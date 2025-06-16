#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查HDF5文件和Qdrant中存储的arXiv论文嵌入向量的脚本。
通过与在线TEI服务重新生成的嵌入向量进行比较来进行验证。
"""

import argparse
import json
import h5py
import numpy as np
import requests
import time
import random
import logging
import os
from datetime import datetime
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

def setup_logger(log_dir, log_level=logging.INFO, script_name="check_embeddings"):
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")
    
    logger = logging.getLogger(script_name)
    logger.setLevel(log_level)
    
    if logger.handlers:
        logger.handlers.clear()
            
    file_handler = logging.FileHandler(log_file)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(file_format)
    logger.addHandler(console_handler)
    
    return logger

def call_tei_service_sync(text, tei_url, prompt_name=None, max_retries=3, retry_delay=1, is_normalize=False):
    """同步调用TEI服务获取文本嵌入向量 (返回fp16)"""
    payload = {"inputs": text, "normalize": is_normalize}
    if prompt_name:
        payload['inputs'] = prompt_name + " " + text
    
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(tei_url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                embedding = np.array(response.json()[0], dtype=np.float16)
                return embedding
            elif response.status_code == 429:
                logging.warning(f"TEI服务返回429 (请求过快)，第 {attempt + 1} 次重试...")
                time.sleep(retry_delay * (2 ** attempt))
                continue
            else:
                logging.error(f"TEI服务错误，状态码: {response.status_code}, 响应: {response.text}")
                return None
                
        except requests.RequestException as e:
            logging.error(f"调用TEI服务失败 (第 {attempt + 1} 次尝试): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
            else:
                return None
    
    logging.error(f"达到最大重试次数 ({max_retries})，无法获取嵌入: {text[:100]}...")
    return None

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    vec1 = vec1.astype(np.float32)
    vec2 = vec2.astype(np.float32)
    
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
        
    return dot_product / (norm_vec1 * norm_vec2)

def generate_sampling_indices(total_count, num_samples, strategy="random", decay_strength=0.8):
    """根据指定策略生成采样索引"""
    if strategy == "random":
        return sorted(random.sample(range(total_count), num_samples))
    
    elif strategy == "exponential_decay":
        decay_rate = decay_strength * 5
        weights = []
        for i in range(total_count):
            position_from_end = total_count - 1 - i
            weight = np.exp(-decay_rate * position_from_end / total_count)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        indices = np.random.choice(total_count, size=num_samples, replace=False, p=weights)
        return sorted(indices.tolist())
    
    elif strategy == "linear_decay":
        weights = []
        for i in range(total_count):
            position_from_end = total_count - 1 - i
            weight = 1.0 - decay_strength * position_from_end / total_count
            weight = max(weight, 0.01)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        indices = np.random.choice(total_count, size=num_samples, replace=False, p=weights)
        return sorted(indices.tolist())
    
    else:
        raise ValueError(f"未知的采样策略: {strategy}")

def connect_to_qdrant(qdrant_url, qdrant_api_key=None):
    """连接到Qdrant服务"""
    if not QDRANT_AVAILABLE:
        raise ImportError("Qdrant客户端未安装。请运行: pip install qdrant-client")
    
    try:
        if qdrant_api_key:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            client = QdrantClient(url=qdrant_url)
        
        collections = client.get_collections()
        return client
    except Exception as e:
        raise ConnectionError(f"无法连接到Qdrant服务 {qdrant_url}: {str(e)}")

def load_metadata_dict(metadata_file, sampled_paper_ids):
    """加载原始元数据并构建字典"""
    metadata_dict = {}
    found_count = 0
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                if found_count == len(sampled_paper_ids):
                    break
                try:
                    paper_data = json.loads(line)
                    paper_id = paper_data.get('id')
                    if paper_id in sampled_paper_ids:
                        metadata_dict[paper_id] = {
                            'title': paper_data.get('title', '').replace('\\n', ' '),
                            'abstract': paper_data.get('abstract', '').replace('\\n', ' ')
                        }
                        if metadata_dict[paper_id]['title'] and metadata_dict[paper_id]['abstract']:
                            found_count += 1
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        raise FileNotFoundError(f"原始元数据文件未找到: {metadata_file}")
    
    return metadata_dict

def check_embeddings_generic(data_source, sampled_paper_ids, metadata_dict, tei_url, prompt_name, logger, source_name):
    """通用的嵌入向量检查函数"""
    logger.info(f"开始检查{source_name}中的嵌入向量...")
    
    consistent_titles = 0
    inconsistent_titles = 0
    failed_title_generation = 0
    title_similarities = []
    
    consistent_abstracts = 0
    inconsistent_abstracts = 0
    failed_abstract_generation = 0
    abstract_similarities = []
    
    processed_papers = 0
    
    for paper_id in sampled_paper_ids:
        if paper_id not in data_source:
            logger.warning(f"在{source_name}中未找到论文 {paper_id}")
            continue
        
        paper_data = data_source[paper_id]
        
        # 根据数据源获取文本内容
        if source_name == "Qdrant":
            # 从Qdrant payload中获取文本
            title_text = paper_data.get('title', {}).get('text', '') if 'title' in paper_data else ''
            abstract_text = paper_data.get('abstract', {}).get('text', '') if 'abstract' in paper_data else ''
        else:
            # HDF5模式，需要从metadata_dict获取文本
            if paper_id not in metadata_dict:
                logger.warning(f"在原始元数据中未找到论文 {paper_id}")
                continue
            paper_metadata = metadata_dict[paper_id]
            title_text = paper_metadata['title']
            abstract_text = paper_metadata['abstract']
        
        processed_papers += 1
        logger.info(f"正在检查论文: {paper_id} ({title_text[:50]}...)")
        
        # 检查标题
        if 'title' in paper_data and title_text:
            if source_name == 'Qdrant':
                # 因为qdrant对于Cosine相似度计算，会自动进行归一化
                new_title_emb = call_tei_service_sync(title_text, tei_url, prompt_name, is_normalize=True)
            else:
                new_title_emb = call_tei_service_sync(title_text, tei_url, prompt_name)
            if new_title_emb is not None:
                if source_name == "Qdrant":
                    stored_title_emb = paper_data['title']['vector']
                else:  # HDF5
                    stored_title_emb = paper_data['h5_title_emb']
                
                similarity = cosine_similarity(stored_title_emb, new_title_emb)
                title_similarities.append(similarity)
                logger.info(f"  标题 - {source_name} vs 新生成: 相似度 = {similarity:.4f}")
                if similarity >= 0.98:
                    consistent_titles += 1
                else:
                    inconsistent_titles += 1
                    logger.warning(f"  标题不一致! ID: {paper_id}, 相似度: {similarity:.4f}")
            else:
                failed_title_generation += 1
        
        # 检查摘要
        if 'abstract' in paper_data and abstract_text:
            if source_name == 'Qdrant':
                new_abstract_emb = call_tei_service_sync(abstract_text, tei_url, prompt_name, is_normalize=True)
            else:
                new_abstract_emb = call_tei_service_sync(abstract_text, tei_url, prompt_name)
            if new_abstract_emb is not None:
                if source_name == "Qdrant":
                    stored_abstract_emb = paper_data['abstract']['vector']
                else:  # HDF5
                    stored_abstract_emb = paper_data['h5_abstract_emb']
                
                similarity = cosine_similarity(stored_abstract_emb, new_abstract_emb)
                abstract_similarities.append(similarity)
                logger.info(f"  摘要 - {source_name} vs 新生成: 相似度 = {similarity:.4f}")
                if similarity >= 0.98:
                    consistent_abstracts += 1
                else:
                    inconsistent_abstracts += 1
                    logger.warning(f"  摘要不一致! ID: {paper_id}, 相似度: {similarity:.4f}")
            else:
                failed_abstract_generation += 1
    
    return {
        'processed_papers': processed_papers,
        'title_stats': {
            'consistent': consistent_titles,
            'inconsistent': inconsistent_titles,
            'failed': failed_title_generation,
            'similarities': title_similarities
        },
        'abstract_stats': {
            'consistent': consistent_abstracts,
            'inconsistent': inconsistent_abstracts,
            'failed': failed_abstract_generation,
            'similarities': abstract_similarities
        }
    }

def print_results(results, source_name, logger):
    """打印检查结果"""
    logger.info("="*50)
    logger.info(f"{source_name}嵌入向量检查结果:")
    logger.info(f"实际处理并比较的论文数: {results['processed_papers']}")
    logger.info("="*50)
    
    # 标题结果
    title_stats = results['title_stats']
    logger.info("标题嵌入检查结果:")
    logger.info(f"  一致数量 (相似度 >= 0.98): {title_stats['consistent']}")
    logger.info(f"  不一致数量 (相似度 < 0.98): {title_stats['inconsistent']}")
    logger.info(f"  生成失败数量: {title_stats['failed']}")
    
    if title_stats['similarities']:
        title_sim_np = np.array(title_stats['similarities'])
        logger.info("  标题相似度统计:")
        logger.info(f"    最小值: {np.min(title_sim_np):.4f}")
        logger.info(f"    最大值: {np.max(title_sim_np):.4f}")
        logger.info(f"    平均值: {np.mean(title_sim_np):.4f}")
        logger.info(f"    中位数: {np.median(title_sim_np):.4f}")
        logger.info(f"    标准差: {np.std(title_sim_np):.4f}")
    
    # 摘要结果
    abstract_stats = results['abstract_stats']
    logger.info("摘要嵌入检查结果:")
    logger.info(f"  一致数量 (相似度 >= 0.98): {abstract_stats['consistent']}")
    logger.info(f"  不一致数量 (相似度 < 0.98): {abstract_stats['inconsistent']}")
    logger.info(f"  生成失败数量: {abstract_stats['failed']}")
    
    if abstract_stats['similarities']:
        abs_sim_np = np.array(abstract_stats['similarities'])
        logger.info("  摘要相似度统计:")
        logger.info(f"    最小值: {np.min(abs_sim_np):.4f}")
        logger.info(f"    最大值: {np.max(abs_sim_np):.4f}")
        logger.info(f"    平均值: {np.mean(abs_sim_np):.4f}")
        logger.info(f"    中位数: {np.median(abs_sim_np):.4f}")
        logger.info(f"    标准差: {np.std(abs_sim_np):.4f}")

def get_sampled_data_from_qdrant(client, collection_name, num_samples, logger):
    """从Qdrant集合中直接采样获取完整数据，包括向量和文本"""
    logger.info(f"开始从Qdrant集合 {collection_name} 中采样 {num_samples} 个论文的完整数据...")
    
    # 首先获取集合的总点数
    try:
        collection_info = client.get_collection(collection_name)
        total_points = collection_info.points_count
        logger.info(f"Qdrant集合中总点数: {total_points}")
        
        if num_samples > total_points:
            num_samples = total_points
            logger.warning(f"请求的采样数量超过总数，调整为: {num_samples}")
            
    except Exception as e:
        logger.warning(f"无法获取集合信息: {str(e)}，将使用滚动方式采样")
        total_points = None
    
    sampled_data = {}  # {paper_id: {title: {vector, text}, abstract: {vector, text}}}
    batch_size = 100
    offset = None
    
    # 使用顺序采样策略（更简单可靠）
    logger.info("使用顺序采样方式")
    processed_points = 0
    max_points_to_process = num_samples * 5  # 限制处理的点数
    
    while len(sampled_data) < num_samples:
            try:
                scroll_result = client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    logger.info("没有更多数据点")
                    break
                    
                for point in points:
                    processed_points += 1
                    paper_id = point.payload.get('paper_id')
                    
                    if paper_id and hasattr(point.vector, 'keys'):
                        # 多向量格式：每个点包含title和abstract向量
                        sampled_data[paper_id] = {
                            'title': {
                                'vector': np.array(point.vector.get('title', []), dtype=np.float16),
                                'text': point.payload.get('title', ''),
                                'point_id': point.id
                            },
                            'abstract': {
                                'vector': np.array(point.vector.get('abstract', []), dtype=np.float16),
                                'text': point.payload.get('abstract', ''),
                                'point_id': point.id
                            }
                        }
                        
                        if len(sampled_data) >= num_samples:
                            logger.info(f"已收集到足够的论文数据: {len(sampled_data)}")
                            break
                    
                    if processed_points >= max_points_to_process:
                        logger.info(f"达到最大处理点数限制: {max_points_to_process}")
                        break
                
                if next_offset is None or len(sampled_data) >= num_samples:
                    break
                offset = next_offset
                
                logger.info(f"已采样 {len(sampled_data)} 个论文，已处理 {processed_points} 个点...")
                
            except Exception as e:
                logger.error(f"获取数据时出错: {str(e)}")
                break
    
    logger.info(f"最终采样到 {len(sampled_data)} 个论文的完整数据")
    return sampled_data

def main():
    parser = argparse.ArgumentParser(description="检查arXiv嵌入向量的脚本")
    parser.add_argument("--h5_file", type=str,
                        help="存储嵌入向量的HDF5文件路径")
    parser.add_argument("--original_metadata_file", type=str,
                        help="原始arXiv元数据JSON文件路径（仅在检查HDF5时需要）")
    parser.add_argument("--tei_url", type=str, default="http://127.0.0.1:8080/embed",
                        help="TEI服务URL")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="要抽样检查的论文数量")
    parser.add_argument("--prompt_name", type=str, default=None,
                        help="TEI服务的提示名称")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="日志输出目录")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    parser.add_argument("--sampling_strategy", type=str, default="random",
                        choices=["random", "exponential_decay", "linear_decay"],
                        help="采样策略")
    parser.add_argument("--decay_strength", type=float, default=0.8,
                        help="衰减采样策略的衰减强度 (0-1)")
    
    # Qdrant相关参数
    parser.add_argument("--qdrant_url", type=str, default="http://127.0.0.1:6333",
                        help="Qdrant服务URL")
    parser.add_argument("--qdrant_api_key", type=str,
                        help="Qdrant API密钥")
    parser.add_argument("--qdrant_collection", type=str,
                        help="Qdrant集合名称")
    parser.add_argument("--check_mode", type=str, default="h5",
                        choices=["h5", "qdrant", "both"],
                        help="检查模式: h5=仅检查HDF5, qdrant=仅检查Qdrant, both=同时检查两者")
    
    args = parser.parse_args()
    
    # 验证参数
    if args.check_mode in ["h5", "both"]:
        if not args.h5_file:
            parser.error("当check_mode为'h5'或'both'时，必须提供--h5_file参数")
        if not args.original_metadata_file:
            parser.error("当check_mode为'h5'或'both'时，必须提供--original_metadata_file参数")
    
    if args.check_mode in ["qdrant", "both"]:
        if not args.qdrant_url or not args.qdrant_collection:
            parser.error("当check_mode为'qdrant'或'both'时，必须提供--qdrant_url和--qdrant_collection参数")

    # 设置日志
    log_level_enum = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logger(args.log_dir, log_level_enum)

    logger.info("开始嵌入向量检查...")
    logger.info(f"检查模式: {args.check_mode}")
    if args.h5_file:
        logger.info(f"HDF5 文件: {args.h5_file}")
    if args.qdrant_url:
        logger.info(f"Qdrant URL: {args.qdrant_url}")
        logger.info(f"Qdrant 集合: {args.qdrant_collection}")
    logger.info(f"原始元数据文件: {args.original_metadata_file}")
    logger.info(f"TEI URL: {args.tei_url}")
    logger.info(f"抽样数量: {args.num_samples}")
    logger.info(f"采样策略: {args.sampling_strategy}")
    if args.sampling_strategy in ["exponential_decay", "linear_decay"]:
        logger.info(f"衰减强度: {args.decay_strength}")

    try:
        # 确定采样来源和生成采样索引
        if args.check_mode in ["h5", "both"]:
            with h5py.File(args.h5_file, 'r') as hf:
                num_total_papers = len(hf['paper_ids'])
                logger.info(f"HDF5文件中总论文数: {num_total_papers}")
                
                if args.num_samples > num_total_papers:
                    args.num_samples = num_total_papers
                
                sampled_indices = generate_sampling_indices(
                    num_total_papers, args.num_samples, 
                    args.sampling_strategy, args.decay_strength
                )
                
                sampled_paper_ids = [hf['paper_ids'][i].decode('utf-8') for i in sampled_indices]
        else:
            # 仅Qdrant模式：从Qdrant集合中直接获取完整的采样数据
            qdrant_client = connect_to_qdrant(args.qdrant_url, args.qdrant_api_key)
            logger.info("成功连接到Qdrant服务")
            
            # 直接从Qdrant中采样完整数据，包括向量和文本
            qdrant_sampled_data = get_sampled_data_from_qdrant(qdrant_client, args.qdrant_collection, args.num_samples, logger)
            
            if not qdrant_sampled_data:
                raise ValueError(f"从Qdrant集合 {args.qdrant_collection} 中未找到任何完整数据")
            
            sampled_paper_ids = list(qdrant_sampled_data.keys())
            logger.info(f"从Qdrant中采样了 {len(sampled_paper_ids)} 篇论文")
        
        # 加载原始元数据（仅在需要时）
        metadata_dict = {}
        if args.check_mode in ["h5", "both"]:
            # 只有在检查HDF5时才需要加载原始元数据文件
            metadata_dict = load_metadata_dict(args.original_metadata_file, sampled_paper_ids)
            logger.info(f"从元数据中找到 {len(metadata_dict)} 篇论文的完整信息")
        else:
            # 仅Qdrant模式下，从Qdrant payload中获取文本数据，不需要加载原始元数据
            logger.info("仅Qdrant模式：将从Qdrant payload中获取文本数据，跳过原始元数据加载")
        
        # 检查HDF5
        if args.check_mode in ["h5", "both"]:
            with h5py.File(args.h5_file, 'r') as hf:
                h5_data = {}
                for i, paper_id in enumerate(sampled_paper_ids):
                    if paper_id in metadata_dict:
                        idx = sampled_indices[i]
                        h5_data[paper_id] = {
                            'h5_title_emb': hf['title_embeddings'][idx],
                            'h5_abstract_emb': hf['abstract_embeddings'][idx]
                        }
                
                h5_results = check_embeddings_generic(
                    h5_data, sampled_paper_ids, metadata_dict, 
                    args.tei_url, args.prompt_name, logger, "HDF5"
                )
                print_results(h5_results, "HDF5", logger)
        
        # 检查Qdrant
        if args.check_mode in ["qdrant", "both"]:
            if args.check_mode == "both":
                # both模式下需要重新获取Qdrant数据
                qdrant_client = connect_to_qdrant(args.qdrant_url, args.qdrant_api_key)
                logger.info("成功连接到Qdrant服务")
                qdrant_data = get_sampled_data_from_qdrant(qdrant_client, args.qdrant_collection, args.num_samples, logger)
                logger.info(f"从Qdrant中获取了 {len(qdrant_data)} 篇论文的数据")
            else:
                # 仅qdrant模式，直接使用之前获取的数据
                qdrant_data = qdrant_sampled_data
                logger.info(f"使用已采样的 {len(qdrant_data)} 篇论文数据")
            
            qdrant_results = check_embeddings_generic(
                qdrant_data, sampled_paper_ids, metadata_dict,
                args.tei_url, args.prompt_name, logger, "Qdrant"
            )
            print_results(qdrant_results, "Qdrant", logger)

    except Exception as e:
        logger.error(f"检查过程中发生严重错误: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 