#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
验证使用TEI服务生成的arXiv论文嵌入向量
检查H5文件中的嵌入向量是否与元数据一一对应
通过调用TEI服务重新生成向量并比较相似度来验证
"""

import json
import argparse
import logging
import h5py
import numpy as np
import requests
import time
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_embeddings_h5(h5_file_path):
    """加载H5文件中的嵌入向量和论文ID"""
    with h5py.File(h5_file_path, 'r') as h5f:
        # 加载ID
        paper_ids = [id.decode('utf-8') if isinstance(id, bytes) else id for id in h5f['paper_ids'][:]]
        
        # 加载嵌入向量
        title_embeddings = h5f['title_embeddings'][:]
        abstract_embeddings = h5f['abstract_embeddings'][:]
        
        # 检查文件中是否包含嵌入信息
        embedding_info = None
        if 'embedding_info' in h5f.attrs:
            try:
                embedding_info = json.loads(h5f.attrs['embedding_info'])
            except:
                pass
        
        logging.info(f"加载了 {len(paper_ids)} 篇论文的嵌入向量")
        logging.info(f"标题嵌入维度: {title_embeddings.shape}")
        logging.info(f"摘要嵌入维度: {abstract_embeddings.shape}")
        
    return paper_ids, title_embeddings, abstract_embeddings, embedding_info


def load_metadata(metadata_file_path):
    """加载元数据文件"""
    with open(metadata_file_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    id_to_index = metadata.get('id_to_index', {})
    papers = metadata.get('papers', [])
    embedding_info = metadata.get('embedding_info', {})
    
    logging.info(f"加载了 {len(papers)} 篇论文的元数据")
    if embedding_info:
        logging.info(f"嵌入信息: {embedding_info}")
    
    return papers, id_to_index, embedding_info


def verify_ids_match(h5_ids, metadata_ids):
    """验证H5文件中的ID和元数据中的ID是否一致"""
    h5_ids_set = set(h5_ids)
    metadata_ids_set = set(metadata_ids)
    
    if h5_ids_set == metadata_ids_set:
        logging.info("ID完全匹配 ✓")
        return True
    else:
        in_h5_not_in_meta = h5_ids_set - metadata_ids_set
        in_meta_not_in_h5 = metadata_ids_set - h5_ids_set
        
        if in_h5_not_in_meta:
            logging.error(f"H5文件中有 {len(in_h5_not_in_meta)} 个ID在元数据中不存在")
            logging.debug(f"样例: {list(in_h5_not_in_meta)[:5]}")
        
        if in_meta_not_in_h5:
            logging.error(f"元数据中有 {len(in_meta_not_in_h5)} 个ID在H5文件中不存在")
            logging.debug(f"样例: {list(in_meta_not_in_h5)[:5]}")
        
        return False


def verify_order_match(h5_ids, metadata_papers, id_to_index):
    """验证H5文件中的ID顺序与元数据中的索引是否一致"""
    all_match = True
    mismatches = 0
    
    for h5_idx, paper_id in enumerate(h5_ids):
        if paper_id in id_to_index:
            meta_idx = id_to_index[paper_id]
            if h5_idx != meta_idx:
                all_match = False
                mismatches += 1
                if mismatches <= 5:  # 只显示前5个不匹配的例子
                    logging.error(f"顺序不匹配: ID {paper_id} 在H5中索引为 {h5_idx}，在元数据中索引为 {meta_idx}")
    
    if all_match:
        logging.info("ID顺序完全匹配 ✓")
    else:
        logging.error(f"共有 {mismatches} 个ID顺序不匹配")
    
    return all_match


def call_tei_service(text, tei_url, prompt_name=None, max_retries=3, retry_delay=1, normalize=True):
    """
    调用TEI服务获取文本嵌入向量
    
    参数:
        text: 要编码的文本
        tei_url: TEI服务的URL
        prompt_name: 可选的提示名称
        max_retries: 最大重试次数
        retry_delay: 重试间隔(秒)
        normalize: 是否对嵌入向量进行归一化
        
    返回:
        numpy数组形式的嵌入向量
    """
    payload = {
        "inputs": text,
        # "normalize": normalize,
        # "truncate": True,
        # "truncation_direction": "right"
    }

    if prompt_name:
        payload['inputs'] = prompt_name + " " + text
    
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(tei_url, json=payload, headers=headers)
            
            if response.status_code == 200:
                # 成功获取嵌入向量
                embedding = np.array(response.json()[0])
                return embedding
            elif response.status_code == 413:
                # 批处理大小错误
                error_data = response.json()
                logging.error(f"批处理大小错误: {error_data}")
                raise ValueError(f"TEI服务批处理大小错误: {error_data}")
            elif response.status_code == 422:
                # 分词错误
                error_data = response.json()
                logging.error(f"分词错误: {error_data}")
                raise ValueError(f"TEI服务分词错误: {error_data}")
            elif response.status_code == 424:
                # 嵌入错误
                error_data = response.json()
                logging.error(f"嵌入错误: {error_data}")
                raise ValueError(f"TEI服务嵌入错误: {error_data}")
            elif response.status_code == 429:
                # 模型过载
                error_data = response.json()
                logging.warning(f"模型过载，尝试重试 ({attempt+1}/{max_retries}): {error_data}")
                time.sleep(retry_delay * (2 ** attempt))  # 指数退避
                continue
            else:
                # 其他错误
                logging.error(f"TEI服务错误，状态码: {response.status_code}, 响应: {response.text}")
                raise ValueError(f"TEI服务返回错误，状态码: {response.status_code}")
                
        except (requests.RequestException, json.JSONDecodeError) as e:
            logging.warning(f"请求失败，尝试重试 ({attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))  # 指数退避
            else:
                raise ValueError(f"调用TEI服务失败: {str(e)}")
    
    raise ValueError(f"达到最大重试次数 ({max_retries})，无法获取嵌入向量")


def verify_embeddings_tei(
    h5_file_path, 
    metadata_file_path, 
    tei_url, 
    num_samples=10, 
    prompt_name=None,
    max_retries=3,
    retry_delay=1
):
    """验证TEI生成的嵌入向量与对应的文本内容是否一致"""
    # 加载嵌入向量和ID
    h5_ids, title_embeddings, abstract_embeddings, h5_embedding_info = load_embeddings_h5(h5_file_path)
    
    # 加载元数据
    papers, id_to_index, metadata_embedding_info = load_metadata(metadata_file_path)
    metadata_ids = [paper['id'] for paper in papers]
    
    # 获取元数据中的prompt_name（如果存在）
    if not prompt_name and 'prompt_name' in metadata_embedding_info:
        prompt_name = metadata_embedding_info['prompt_name']
        logging.info(f"从元数据中获取到prompt_name: {prompt_name}")
    
    # 测试TEI服务连接
    test_text = "测试TEI服务连接"
    try:
        test_embedding = call_tei_service(test_text, tei_url, prompt_name, max_retries, retry_delay)
        embedding_dim = test_embedding.shape[0]
        logging.info(f"TEI服务连接成功，嵌入维度: {embedding_dim}")
    except Exception as e:
        logging.error(f"TEI服务连接测试失败: {str(e)}")
        logging.error("请确保TEI服务正在运行，并且可以通过指定的URL访问")
        return False
    
    # 验证ID是否匹配
    ids_match = verify_ids_match(h5_ids, metadata_ids)
    
    # 验证ID顺序是否匹配
    order_match = verify_order_match(h5_ids, papers, id_to_index)
    
    # 随机选择样本进行嵌入向量验证
    if num_samples > len(h5_ids):
        num_samples = len(h5_ids)
    
    sample_indices = np.random.choice(len(h5_ids), num_samples, replace=False)
    
    title_cos_sims = []
    abstract_cos_sims = []
    
    logging.info(f"正在验证 {num_samples} 个随机样本的嵌入向量...")
    
    for idx in tqdm(sample_indices):
        paper_id = h5_ids[idx]
        paper_idx = id_to_index.get(paper_id)
        
        if paper_idx is None:
            logging.error(f"在元数据中找不到ID: {paper_id}")
            continue
        
        paper = papers[paper_idx]
        
        # 从H5文件获取存储的嵌入向量
        stored_title_emb = title_embeddings[idx]
        stored_abstract_emb = abstract_embeddings[idx]
        
        # 重新计算嵌入向量
        title_text = paper['title']
        abstract_text = paper['abstract']
        
        try:
            # 使用TEI服务重新计算嵌入向量
            computed_title_emb = call_tei_service(title_text, tei_url, prompt_name, max_retries, retry_delay)
            computed_abstract_emb = call_tei_service(abstract_text, tei_url, prompt_name, max_retries, retry_delay)
            
            # 计算余弦相似度
            title_cos_sim = np.dot(stored_title_emb, computed_title_emb) / (
                np.linalg.norm(stored_title_emb) * np.linalg.norm(computed_title_emb)
            )
            
            abstract_cos_sim = np.dot(stored_abstract_emb, computed_abstract_emb) / (
                np.linalg.norm(stored_abstract_emb) * np.linalg.norm(computed_abstract_emb)
            )
            
            title_cos_sims.append(title_cos_sim)
            abstract_cos_sims.append(abstract_cos_sim)
            
        except Exception as e:
            logging.error(f"处理论文 {paper_id} 时出错: {str(e)}")
    
    # 输出验证结果
    if not title_cos_sims or not abstract_cos_sims:
        logging.error("没有成功验证任何样本，无法计算相似度")
        return False
    
    avg_title_sim = np.mean(title_cos_sims)
    avg_abstract_sim = np.mean(abstract_cos_sims)
    
    logging.info(f"标题嵌入向量平均余弦相似度: {avg_title_sim:.4f}")
    logging.info(f"摘要嵌入向量平均余弦相似度: {avg_abstract_sim:.4f}")
    
    # TEI服务可能有轻微差异，设置合理的阈值
    sim_threshold = 0.95
        
    if avg_title_sim > sim_threshold and avg_abstract_sim > sim_threshold:
        logging.info("嵌入向量验证通过 ✓")
        embedding_match = True
    else:
        logging.warning(f"嵌入向量相似度低于预期 (阈值: {sim_threshold})")
        logging.warning("这可能是由于TEI服务配置不同、随机性或其他因素导致的")
        embedding_match = False
    
    # 输出总体验证结果
    if ids_match and order_match and embedding_match:
        logging.info("="*50)
        logging.info("验证通过！H5文件中的嵌入向量与元数据完全对应 ✓")
        logging.info("="*50)
        return True
    else:
        logging.warning("="*50)
        logging.warning("验证发现问题！请检查上述错误信息")
        logging.warning("="*50)
        return False


def main():
    parser = argparse.ArgumentParser(description="验证使用TEI服务生成的arXiv论文嵌入向量")
    parser.add_argument("--h5_file", type=str, required=True,
                        help="H5文件路径，包含嵌入向量")
    parser.add_argument("--metadata_file", type=str, required=True,
                        help="元数据JSON文件路径")
    parser.add_argument("--tei_url", type=str, default="http://127.0.0.1:8080/embed",
                        help="TEI服务URL")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="要验证的随机样本数量")
    parser.add_argument("--prompt_name", type=str, default=None,
                       help="TEI服务的提示名称，如果为None则从元数据中读取")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="调用TEI服务的最大重试次数")
    parser.add_argument("--retry_delay", type=int, default=1,
                        help="调用TEI服务的重试间隔（秒）")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 显示配置信息
    logging.info("="*50)
    logging.info("使用TEI服务生成的arXiv论文嵌入向量验证")
    logging.info("="*50)
    for arg, value in sorted(vars(args).items()):
        logging.info(f"参数 {arg}: {value}")
    logging.info("="*50)
    
    # 验证嵌入向量
    verify_embeddings_tei(
        args.h5_file,
        args.metadata_file,
        args.tei_url,
        args.num_samples,
        args.prompt_name,
        args.max_retries,
        args.retry_delay
    )


if __name__ == "__main__":
    main() 