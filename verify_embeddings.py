#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
验证arXiv论文嵌入向量文件
检查H5文件中的嵌入向量是否与元数据一一对应
"""

import os
import json
import argparse
import logging
import h5py
import numpy as np
from sentence_transformers import SentenceTransformer
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
        
        logging.info(f"加载了 {len(paper_ids)} 篇论文的嵌入向量")
        logging.info(f"标题嵌入维度: {title_embeddings.shape}")
        logging.info(f"摘要嵌入维度: {abstract_embeddings.shape}")
        
    return paper_ids, title_embeddings, abstract_embeddings


def load_metadata(metadata_file_path):
    """加载元数据文件"""
    with open(metadata_file_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    id_to_index = metadata.get('id_to_index', {})
    papers = metadata.get('papers', [])
    
    logging.info(f"加载了 {len(papers)} 篇论文的元数据")
    
    return papers, id_to_index


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


def verify_embeddings(
    h5_file_path, 
    metadata_file_path, 
    model_path, 
    num_samples=10, 
    batch_size=2, 
    use_flash_attention=False
):
    """验证嵌入向量与对应的文本内容是否一致"""
    # 加载模型
    logging.info(f"加载模型: {model_path}")
    try:
        if use_flash_attention:
            model = SentenceTransformer(model_path, use_flash_attention=True)
            logging.info("已启用Flash Attention")
        else:
            model = SentenceTransformer(model_path)
    except Exception as e:
        logging.error(f"加载模型失败: {e}")
        return False
    
    # 加载嵌入向量和ID
    h5_ids, title_embeddings, abstract_embeddings = load_embeddings_h5(h5_file_path)
    
    # 加载元数据
    papers, id_to_index = load_metadata(metadata_file_path)
    metadata_ids = [paper['id'] for paper in papers]
    
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
        
        computed_title_emb = model.encode([title_text], batch_size=batch_size)[0]
        computed_abstract_emb = model.encode([abstract_text], batch_size=batch_size)[0]
        
        # 计算余弦相似度
        title_cos_sim = np.dot(stored_title_emb, computed_title_emb) / (
            np.linalg.norm(stored_title_emb) * np.linalg.norm(computed_title_emb)
        )
        
        abstract_cos_sim = np.dot(stored_abstract_emb, computed_abstract_emb) / (
            np.linalg.norm(stored_abstract_emb) * np.linalg.norm(computed_abstract_emb)
        )
        
        title_cos_sims.append(title_cos_sim)
        abstract_cos_sims.append(abstract_cos_sim)
    
    # 输出验证结果
    avg_title_sim = np.mean(title_cos_sims)
    avg_abstract_sim = np.mean(abstract_cos_sims)
    
    logging.info(f"标题嵌入向量平均余弦相似度: {avg_title_sim:.4f}")
    logging.info(f"摘要嵌入向量平均余弦相似度: {avg_abstract_sim:.4f}")
    
    # 一般来说，相同的输入应该产生相同的嵌入向量，所以相似度应该非常接近1
    if avg_title_sim > 0.99 and avg_abstract_sim > 0.99:
        logging.info("嵌入向量验证通过 ✓")
        embedding_match = True
    else:
        logging.warning("嵌入向量相似度低于预期")
        logging.warning("这可能是由于模型加载参数不同、随机性或其他因素导致的")
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
    parser = argparse.ArgumentParser(description="验证arXiv论文嵌入向量文件")
    parser.add_argument("--h5_file", type=str, required=True,
                        help="H5文件路径，包含嵌入向量")
    parser.add_argument("--metadata_file", type=str, required=True,
                        help="元数据JSON文件路径")
    parser.add_argument("--model_path", type=str, required=True,
                        help="用于生成嵌入向量的模型路径，必须与原始模型相同")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="要验证的随机样本数量")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="验证时的批处理大小")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    parser.add_argument("--use_flash_attention", action="store_true",
                        help="启用Flash Attention加速（如果可用）")
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 显示配置信息
    logging.info("="*50)
    logging.info("arXiv论文嵌入向量验证")
    logging.info("="*50)
    for arg, value in sorted(vars(args).items()):
        logging.info(f"参数 {arg}: {value}")
    logging.info("="*50)
    
    # 验证嵌入向量
    verify_embeddings(
        args.h5_file,
        args.metadata_file,
        args.model_path,
        args.num_samples,
        args.batch_size,
        args.use_flash_attention
    )


if __name__ == "__main__":
    main() 