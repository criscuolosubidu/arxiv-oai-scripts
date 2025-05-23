#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查HDF5文件中存储的arXiv论文嵌入向量的脚本。
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

def setup_logger(log_dir, log_level=logging.INFO, script_name="check_embeddings"):
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")
    
    logger = logging.getLogger(script_name) # Use a named logger
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers if called multiple times
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

# 从 generate_embeddings_tei.py 复制并适配
def call_tei_service_sync(text, tei_url, prompt_name=None, max_retries=3, retry_delay=1):
    """同步调用TEI服务获取文本嵌入向量 (返回fp16)"""
    payload = {"inputs": text, "normalize": False}
    if prompt_name:
        payload['inputs'] = prompt_name + " " + text
    
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(tei_url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                embedding = np.array(response.json()[0], dtype=np.float16) # 确保是 fp16
                return embedding
            elif response.status_code == 429:
                logging.warning(f"TEI服务返回429 (请求过快)，第 {attempt + 1} 次重试...")
                time.sleep(retry_delay * (2 ** attempt))
                continue
            else:
                logging.error(f"TEI服务错误，状态码: {response.status_code}, 响应: {response.text}")
                # 不直接抛出 ValueError，而是返回 None，让主逻辑处理
                return None
                
        except requests.RequestException as e:
            logging.error(f"调用TEI服务失败 (第 {attempt + 1} 次尝试): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
            else:
                # 达到最大重试次数后返回 None
                return None
    
    logging.error(f"达到最大重_TEI调用重试次数 ({max_retries})，无法获取嵌入: {text[:100]}...")
    return None

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    # 确保向量是浮点数类型，以避免整数运算问题
    vec1 = vec1.astype(np.float32)
    vec2 = vec2.astype(np.float32)
    
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0  # 避免除以零
        
    return dot_product / (norm_vec1 * norm_vec2)

def main():
    parser = argparse.ArgumentParser(description="检查arXiv嵌入向量的脚本")
    parser.add_argument("--h5_file", type=str, required=True,
                        help="存储嵌入向量的HDF5文件路径")
    parser.add_argument("--original_metadata_file", type=str, required=True,
                        help="原始arXiv元数据JSON文件路径 (例如 arxiv-metadata-oai-snapshot.json)")
    parser.add_argument("--tei_url", type=str, default="http://127.0.0.1:8080/embed",
                        help="TEI服务URL")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="要抽样检查的论文数量")
    parser.add_argument("--prompt_name", type=str, default=None,
                        help="TEI服务的提示名称 (如果生成时使用过)")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="检查脚本的日志输出目录")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    
    args = parser.parse_args()

    # 设置日志
    log_level_enum = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logger(args.log_dir, log_level_enum)

    logger.info("开始嵌入向量检查...")
    logger.info(f"HDF5 文件: {args.h5_file}")
    logger.info(f"原始元数据文件: {args.original_metadata_file}")
    logger.info(f"TEI URL: {args.tei_url}")
    logger.info(f"抽样数量: {args.num_samples}")
    if args.prompt_name:
        logger.info(f"Prompt Name: {args.prompt_name}")

    try:
        with h5py.File(args.h5_file, 'r') as hf:
            if 'paper_ids' not in hf or \
               'title_embeddings' not in hf or \
               'abstract_embeddings' not in hf:
                logger.error("HDF5文件缺少必要的数据集 (paper_ids, title_embeddings, abstract_embeddings)")
                return

            num_total_papers = len(hf['paper_ids'])
            logger.info(f"HDF5文件中总论文数: {num_total_papers}")

            if args.num_samples <= 0:
                logger.info("抽样数量为0，不执行检查。")
                return
            if args.num_samples > num_total_papers:
                logger.warning(f"抽样数量 ({args.num_samples}) 大于总论文数 ({num_total_papers})，将检查所有论文。")
                args.num_samples = num_total_papers
            
            # 生成随机抽样索引
            sampled_indices = sorted(random.sample(range(num_total_papers), args.num_samples))
            
            logger.info(f"将从以下索引抽样: {sampled_indices}")

            # 获取抽样数据
            sampled_paper_ids = [hf['paper_ids'][i].decode('utf-8') for i in sampled_indices] # ID是vlen str
            sampled_title_embeddings_h5 = hf['title_embeddings'][sampled_indices]
            sampled_abstract_embeddings_h5 = hf['abstract_embeddings'][sampled_indices]

        # 构建一个字典来存储抽样论文的HDF5数据，并准备填充原始元数据
        papers_to_check = {}
        for i, paper_id in enumerate(sampled_paper_ids):
            papers_to_check[paper_id] = {
                'h5_title_emb': sampled_title_embeddings_h5[i],
                'h5_abstract_emb': sampled_abstract_embeddings_h5[i],
                'original_title': None,
                'original_abstract': None
            }
        
        logger.info(f"已加载 {len(sampled_paper_ids)} 个样本的HDF5数据。")
        logger.info("现在从原始元数据文件中查找对应的标题和摘要...")

        # 读取原始元数据文件，填充标题和摘要
        found_count = 0
        try:
            with open(args.original_metadata_file, 'r', encoding='utf-8') as f_meta:
                for line in f_meta:
                    if found_count == args.num_samples: # 如果都找到了就提前退出
                        break
                    try:
                        paper_data = json.loads(line)
                        paper_id = paper_data.get('id')
                        if paper_id in papers_to_check:
                            papers_to_check[paper_id]['original_title'] = paper_data.get('title', '').replace('\\n', ' ')
                            papers_to_check[paper_id]['original_abstract'] = paper_data.get('abstract', '').replace('\\n', ' ')
                            if papers_to_check[paper_id]['original_title'] and papers_to_check[paper_id]['original_abstract']:
                                found_count +=1
                                logger.debug(f"找到论文 {paper_id} 的元数据。")
                            else:
                                logger.warning(f"论文 {paper_id} 在元数据中缺少标题或摘要。")
                    except json.JSONDecodeError:
                        logger.warning(f"元数据文件有一行JSON解析失败: {line[:100]}...")
                        continue
        except FileNotFoundError:
            logger.error(f"原始元数据文件未找到: {args.original_metadata_file}")
            return
        
        if found_count < args.num_samples:
            logger.warning(f"只在元数据中找到了 {found_count}/{args.num_samples} 篇抽样论文的完整信息。")

        # 开始比较
        consistent_titles = 0
        inconsistent_titles = 0
        failed_title_generation = 0
        title_similarities = []
        
        consistent_abstracts = 0
        inconsistent_abstracts = 0
        failed_abstract_generation = 0
        abstract_similarities = []

        processed_papers = 0

        for paper_id, data in papers_to_check.items():
            if not data['original_title'] or not data['original_abstract']:
                logger.warning(f"跳过论文 {paper_id}，因其缺少原始标题或摘要。")
                continue
            
            processed_papers += 1
            logger.info(f"正在检查论文: {paper_id} ({data['original_title'][:50]}...)")

            # 检查标题
            logger.debug(f"重新生成标题嵌入: {data['original_title'][:50]}...")
            new_title_emb = call_tei_service_sync(data['original_title'], args.tei_url, args.prompt_name)
            if new_title_emb is not None:
                similarity = cosine_similarity(data['h5_title_emb'], new_title_emb)
                title_similarities.append(similarity)
                logger.info(f"  标题 '{data['original_title'][:30]}...' - H5 vs 新生成: 相似度 = {similarity:.4f}")
                if similarity >= 0.98:
                    consistent_titles += 1
                else:
                    inconsistent_titles += 1
                    logger.warning(f"  标题不一致! ID: {paper_id}, 相似度: {similarity:.4f}")
            else:
                logger.error(f"  无法为论文 {paper_id} 生成新的标题嵌入。")
                failed_title_generation +=1

            # 检查摘要
            logger.debug(f"重新生成摘要嵌入: {data['original_abstract'][:50]}...")
            new_abstract_emb = call_tei_service_sync(data['original_abstract'], args.tei_url, args.prompt_name)
            if new_abstract_emb is not None:
                similarity = cosine_similarity(data['h5_abstract_emb'], new_abstract_emb)
                abstract_similarities.append(similarity)
                logger.info(f"  摘要 '{data['original_abstract'][:30]}...' - H5 vs 新生成: 相似度 = {similarity:.4f}")
                if similarity >= 0.98:
                    consistent_abstracts += 1
                else:
                    inconsistent_abstracts += 1
                    logger.warning(f"  摘要不一致! ID: {paper_id}, 相似度: {similarity:.4f}")
            else:
                logger.error(f"  无法为论文 {paper_id} 生成新的摘要嵌入。")
                failed_abstract_generation += 1
        
        # 打印总结
        logger.info("="*50)
        logger.info("嵌入向量检查完成！")
        logger.info(f"总共计划检查抽样论文数: {args.num_samples}")
        logger.info(f"实际处理并比较的论文数: {processed_papers}")
        logger.info("="*50)
        logger.info("标题嵌入检查结果:")
        logger.info(f"  一致数量 (相似度 >= 0.98): {consistent_titles}")
        logger.info(f"  不一致数量 (相似度 < 0.98): {inconsistent_titles}")
        logger.info(f"  生成失败数量: {failed_title_generation}")
        if title_similarities:
            title_sim_np = np.array(title_similarities)
            logger.info("  标题相似度统计:")
            logger.info(f"    最小值: {np.min(title_sim_np):.4f}")
            logger.info(f"    最大值: {np.max(title_sim_np):.4f}")
            logger.info(f"    平均值: {np.mean(title_sim_np):.4f}")
            logger.info(f"    中位数: {np.median(title_sim_np):.4f}")
            logger.info(f"    标准差: {np.std(title_sim_np):.4f}")
            logger.info(f"    25百分位数: {np.percentile(title_sim_np, 25):.4f}")
            logger.info(f"    75百分位数: {np.percentile(title_sim_np, 75):.4f}")
        logger.info("="*50)
        logger.info("摘要嵌入检查结果:")
        logger.info(f"  一致数量 (相似度 >= 0.98): {consistent_abstracts}")
        logger.info(f"  不一致数量 (相似度 < 0.98): {inconsistent_abstracts}")
        logger.info(f"  生成失败数量: {failed_abstract_generation}")
        if abstract_similarities:
            abs_sim_np = np.array(abstract_similarities)
            logger.info("  摘要相似度统计:")
            logger.info(f"    最小值: {np.min(abs_sim_np):.4f}")
            logger.info(f"    最大值: {np.max(abs_sim_np):.4f}")
            logger.info(f"    平均值: {np.mean(abs_sim_np):.4f}")
            logger.info(f"    中位数: {np.median(abs_sim_np):.4f}")
            logger.info(f"    标准差: {np.std(abs_sim_np):.4f}")
            logger.info(f"    25百分位数: {np.percentile(abs_sim_np, 25):.4f}")
            logger.info(f"    75百分位数: {np.percentile(abs_sim_np, 75):.4f}")
        logger.info("="*50)

    except Exception as e:
        logger.error(f"检查过程中发生严重错误: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 