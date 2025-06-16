#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
验证arXiv论文嵌入向量文件
检查H5文件中的嵌入向量是否与元数据一一对应
支持使用 sentence-transformers 或 transformers 生成的向量进行验证
"""

import os
import json
import argparse
import logging
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def last_token_pool(last_hidden_states, attention_mask):
    """
    提取最后一个token的表示作为整个序列的表示
    根据注意力掩码正确处理左填充和右填充
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        # 如果是左填充，直接取最后一个token
        return last_hidden_states[:, -1]
    else:
        # 如果是右填充，需要根据attention_mask来获取每个样本的最后一个有效token
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description, query):
    """
    构建指令格式的查询
    """
    return f'Instruct: {task_description}\nQuery: {query}'


def load_embeddings_h5(h5_file_path):
    """加载H5文件中的嵌入向量和论文ID"""
    with h5py.File(h5_file_path, 'r') as h5f:
        # 加载ID
        paper_ids = [id.decode('utf-8') if isinstance(id, bytes) else id for id in h5f['paper_ids'][:]]
        
        # 加载嵌入向量
        title_embeddings = h5f['title_embeddings'][:]
        abstract_embeddings = h5f['abstract_embeddings'][:]
        
        # 检查文件中是否包含使用transformers的标志
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
    if 'use_transformers' in embedding_info:
        logging.info(f"嵌入向量生成方式: {'transformers' if embedding_info['use_transformers'] else 'sentence-transformers'}")
    
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


def generate_embeddings_transformers(model_data, texts, batch_size=2, max_length=4096, task_description=None):
    """
    使用transformers库批量生成嵌入向量
    """
    tokenizer = model_data['tokenizer']
    model = model_data['model']
    device = next(model.parameters()).device
    
    # 检测是否使用bf16或fp16
    using_bf16 = model.dtype == torch.bfloat16 or model.dtype == torch.float16
    
    embeddings = []
    
    # 处理文本格式
    if task_description:
        # 如果提供了任务描述，构建指令格式的查询
        processed_texts = [get_detailed_instruct(task_description, text) for text in texts]
    else:
        processed_texts = texts
    
    # 批量处理
    for i in range(0, len(processed_texts), batch_size):
        batch_texts = processed_texts[i:i+batch_size]
        
        # 使用tokenizer处理文本
        batch_dict = tokenizer(
            batch_texts, 
            max_length=max_length, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(device)
        
        # 模型推理
        with torch.no_grad():
            if using_bf16:
                # 使用autocast以确保在使用bf16/fp16时不会出现精度问题
                with torch.autocast(device_type=device.type):
                    outputs = model(**batch_dict)
            else:
                outputs = model(**batch_dict)
            
        # 处理模型输出，获取嵌入向量
        batch_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        # 标准化嵌入向量
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        
        # 转为CPU并添加到结果列表
        embeddings.append(batch_embeddings.cpu().numpy())
    
    # 合并所有批次的结果
    embeddings = np.vstack(embeddings)
    
    # 清理GPU缓存，释放内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return embeddings


def verify_embeddings(
    h5_file_path, 
    metadata_file_path, 
    model_path, 
    num_samples=10, 
    batch_size=2, 
    use_flash_attention=False,
    use_transformers=False,
    task_description=None,
    bf16=False,
    model_attn_implementation="eager",
    max_length=4096
):
    """验证嵌入向量与对应的文本内容是否一致"""
    # 加载嵌入向量和ID
    h5_ids, title_embeddings, abstract_embeddings, h5_embedding_info = load_embeddings_h5(h5_file_path)
    
    # 加载元数据
    papers, id_to_index, metadata_embedding_info = load_metadata(metadata_file_path)
    metadata_ids = [paper['id'] for paper in papers]
    
    # 确定是否应该使用transformers或sentence-transformers
    # 首先检查元数据中的信息
    should_use_transformers = metadata_embedding_info.get('use_transformers', False)
    if should_use_transformers != use_transformers:
        logging.warning(f"元数据显示嵌入向量使用了{'transformers' if should_use_transformers else 'sentence-transformers'}，"
                       f"但您指定了{'transformers' if use_transformers else 'sentence-transformers'}。"
                       f"已根据元数据自动调整为{'transformers' if should_use_transformers else 'sentence-transformers'}。")
        use_transformers = should_use_transformers

    # 加载模型
    logging.info(f"加载模型: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        if use_transformers:
            logging.info("使用transformers库加载模型")
            model_kwargs = {}
            if torch.cuda.is_available():
                # 精度设置
                torch_dtype = torch.bfloat16 if bf16 else torch.float32
                model_kwargs["torch_dtype"] = torch_dtype
                
                # 注意力机制设置
                if model_attn_implementation != "eager":
                    model_kwargs["attn_implementation"] = model_attn_implementation
                    if model_attn_implementation == "flash_attention_2":
                        logging.info("使用Flash Attention 2进行加速")
                        # Flash Attention只支持bf16或fp16
                        if not bf16:
                            logging.warning("Flash Attention 2只支持bf16或fp16精度，自动启用bf16精度")
                            model_kwargs["torch_dtype"] = torch.bfloat16
                            bf16 = True
            
            # 加载tokenizer和模型
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path, **model_kwargs)
            model.to(device)
            
            # 将tokenizer和model打包为一个字典，方便传递
            model_data = {
                'tokenizer': tokenizer,
                'model': model
            }
            
            precision_info = "BF16" if bf16 or (model_attn_implementation == "flash_attention_2") else "FP32"
            logging.info(f"已加载transformers模型，使用{precision_info}精度和{model_attn_implementation}注意力机制")
        else:
            logging.info("使用sentence-transformers库加载模型")
            model = SentenceTransformer(model_path)
            model.max_seq_length = max_length
            model.to(device)
            model_data = model
    except Exception as e:
        logging.error(f"加载模型失败: {e}")
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
        
        if use_transformers:
            # 使用transformers计算嵌入向量
            computed_title_emb = generate_embeddings_transformers(
                model_data, 
                [title_text], 
                batch_size=batch_size,
                max_length=max_length,
                task_description=task_description
            )[0]
            
            computed_abstract_emb = generate_embeddings_transformers(
                model_data, 
                [abstract_text], 
                batch_size=batch_size,
                max_length=max_length,
                task_description=task_description
            )[0]
        else:
            # 使用sentence-transformers计算嵌入向量
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
    # 但对于transformers和flash-attention，可能会有一些小的差异
    if use_transformers:
        sim_threshold = 0.95  # transformers可能有轻微差异
    else:
        sim_threshold = 0.99  # sentence-transformers应该几乎相同
        
    if avg_title_sim > sim_threshold and avg_abstract_sim > sim_threshold:
        logging.info("嵌入向量验证通过 ✓")
        embedding_match = True
    else:
        logging.warning(f"嵌入向量相似度低于预期 (阈值: {sim_threshold})")
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
    parser.add_argument("--max_length", type=int, default=4096,
                        help="最大序列长度")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    
    # transformers相关参数
    parser.add_argument("--use_transformers", action="store_true",
                       help="使用transformers库替代sentence-transformers进行验证")
    parser.add_argument("--use_flash_attention", action="store_true",
                       help="启用Flash Attention加速（仅在use_transformers=True时有效）")
    parser.add_argument("--task_description", type=str, default=None,
                       help="任务描述，用于构建指令格式的查询（仅在use_transformers=True时有效）")
    parser.add_argument("--bf16", action="store_true",
                       help="使用BF16精度而不是FP32精度，可以提高性能和减少内存使用（仅在use_transformers=True时有效）")
    parser.add_argument("--model_attn_implementation", type=str, default="eager",
                       choices=["eager", "sdpa", "flash_attention_2"],
                       help="模型注意力实现方式（仅在use_transformers=True时有效）")
    
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
        args.use_flash_attention,
        args.use_transformers,
        args.task_description,
        args.bf16,
        args.model_attn_implementation,
        args.max_length
    )


if __name__ == "__main__":
    main() 