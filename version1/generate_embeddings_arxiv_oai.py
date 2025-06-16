#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成arXiv论文的语义嵌入
使用E5-Mistral-7B模型对arXiv论文的标题和摘要生成嵌入向量
"""

import json
import os
import time
import argparse
import torch
import numpy as np
import logging
import h5py
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# 配置日志
def setup_logger(log_dir, log_level=logging.INFO):
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名，包含时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"embedding_generation_{timestamp}.log")
    
    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 清除现有处理程序
    if logger.handlers:
        logger.handlers.clear()
    
    # 添加文件处理程序
    file_handler = logging.FileHandler(log_file)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # 添加控制台处理程序
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(file_format)
    logger.addHandler(console_handler)
    
    return logger


class ArxivDataset(Dataset):
    """arXiv数据集加载器"""
    
    def __init__(self, file_path, start_idx=0, max_samples=None):
        self.file_path = file_path
        self.data = []
        self.start_idx = start_idx
        self.max_samples = max_samples
        
        # 加载数据
        self._load_data()
        
    def _load_data(self):
        """加载arXiv数据"""
        logging.info(f"正在从 {self.file_path} 加载数据...")
        count = 0
        skipped = 0
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # 跳过开始的行
            for _ in range(self.start_idx):
                next(f, None)
                skipped += 1
                
            for line in f:
                try:
                    paper = json.loads(line)
                    # 确保必要字段存在
                    if 'title' in paper and 'abstract' in paper and paper['title'] and paper['abstract']:
                        self.data.append(paper)
                        count += 1
                        
                        # 达到最大样本数时停止
                        if self.max_samples and count >= self.max_samples:
                            break
                except json.JSONDecodeError:
                    continue
                    
        logging.info(f"加载了 {count} 篇论文数据，跳过了 {skipped} 行")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        paper = self.data[idx]
        return {
            'id': paper.get('id', ''),
            'title': paper.get('title', '').replace('\n', ' '),
            'abstract': paper.get('abstract', '').replace('\n', ' '),
            'authors': paper.get('authors', ''),
            'categories': paper.get('categories', ''),
            'journal_ref': paper.get('journal-ref', ''),
            'doi': paper.get('doi', ''),
            'update_date': paper.get('update_date', '')
        }


def collate_batch(batch):
    """将批次数据整理成所需格式"""
    ids = [item['id'] for item in batch]
    titles = [item['title'] for item in batch]
    abstracts = [item['abstract'] for item in batch]
    metadata = [{
        'authors': item['authors'],
        'categories': item['categories'],
        'journal_ref': item['journal_ref'],
        'doi': item['doi'],
        'update_date': item['update_date']
    } for item in batch]
    
    return {
        'ids': ids,
        'titles': titles,
        'abstracts': abstracts,
        'metadata': metadata
    }


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


def generate_embeddings(model, texts, batch_size=8, prompt_name=None):
    """批量生成嵌入向量 (使用SentenceTransformer)"""
    embeddings = []
    
    # 按批次处理
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        if prompt_name:
            batch_embeddings = model.encode(batch_texts, prompt_name=prompt_name)
        else:
            batch_embeddings = model.encode(batch_texts)
        embeddings.extend(batch_embeddings)
    
    # 清理GPU缓存，释放内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return np.array(embeddings)


def generate_embeddings_transformers(model_data, texts, batch_size=8, max_length=4096, task_description=None):
    """
    使用transformers库批量生成嵌入向量
    
    参数:
        model_data: 包含tokenizer和model的字典
        texts: 待编码的文本列表
        batch_size: 批处理大小
        max_length: 最大序列长度
        task_description: 如果提供，会将文本构建为指令格式
    
    返回:
        numpy数组形式的嵌入向量
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


def process_and_save(
        dataloader, 
        model, 
        output_dir, 
        batch_size=8, 
        save_every=100, 
        start_batch=0, 
        storage_format='h5', 
        numpy_save_interval=10,
        use_transformers=False,
        task_description=None,
        max_length=4096
    ):
    """处理数据并保存嵌入向量和元数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    total_batches = len(dataloader)
    all_ids = []
    all_metadata = []
    
    # 创建文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确定保存格式
    if storage_format == 'h5':
        embedding_file = os.path.join(output_dir, f"arxiv_embeddings_{timestamp}.h5")
        title_emb_dataset = None
        abstract_emb_dataset = None
        h5_file = h5py.File(embedding_file, 'w')
    else:  # numpy格式
        title_embedding_file = os.path.join(output_dir, f"arxiv_title_embeddings_{timestamp}")
        abstract_embedding_file = os.path.join(output_dir, f"arxiv_abstract_embeddings_{timestamp}")
        title_embeddings_list = []
        abstract_embeddings_list = []
        numpy_batch_counter = 0
        numpy_file_index = 0
    
    # 元数据文件
    metadata_file = os.path.join(output_dir, f"arxiv_metadata_{timestamp}.json")
    
    logging.info(f"将向量保存到: {output_dir}")
    logging.info(f"将元数据保存到: {metadata_file}")
    
    start_time = time.time()
    
    try:
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="处理批次")):
            # 跳过已处理的批次
            if batch_idx < start_batch:
                continue
                
            ids = batch['ids']
            titles = batch['titles']
            abstracts = batch['abstracts']
            metadata = batch['metadata']
            
            # 生成嵌入
            if use_transformers:
                title_embeddings = generate_embeddings_transformers(model, titles, batch_size, max_length, task_description)
                abstract_embeddings = generate_embeddings_transformers(model, abstracts, batch_size, max_length, task_description)
            else:
                title_embeddings = generate_embeddings(model, titles, batch_size)
                abstract_embeddings = generate_embeddings(model, abstracts, batch_size)
            
            # 保存数据
            for i in range(len(ids)):
                # 保存ID和元数据
                all_ids.append(ids[i])
                paper_metadata = {
                    'id': ids[i],
                    'title': titles[i],
                    'abstract': abstracts[i],
                    'authors': metadata[i]['authors'],
                    'categories': metadata[i]['categories'],
                    'journal_ref': metadata[i]['journal_ref'],
                    'doi': metadata[i]['doi'],
                    'update_date': metadata[i]['update_date']
                }
                all_metadata.append(paper_metadata)
            
            # 保存嵌入向量
            if storage_format == 'h5':
                # 第一个批次，创建数据集
                if title_emb_dataset is None:
                    embedding_dim = title_embeddings.shape[1]
                    # 预分配空间，设置可扩展的数据集，使用平衡的压缩级别5
                    title_emb_dataset = h5_file.create_dataset(
                        'title_embeddings', 
                        shape=(len(ids), embedding_dim),
                        maxshape=(None, embedding_dim),
                        dtype='float32',
                        compression='gzip', 
                        compression_opts=5
                    )
                    abstract_emb_dataset = h5_file.create_dataset(
                        'abstract_embeddings', 
                        shape=(len(ids), embedding_dim),
                        maxshape=(None, embedding_dim),
                        dtype='float32',
                        compression='gzip', 
                        compression_opts=5
                    )
                    # 存储ID索引，为ID分配足够长的字符串空间
                    id_dtype = h5py.special_dtype(vlen=str)
                    h5_file.create_dataset(
                        'paper_ids', 
                        data=np.array(ids, dtype=object), 
                        maxshape=(None,), 
                        dtype=id_dtype, 
                        chunks=True
                    )
                    
                    # 添加第一个批次的数据
                    title_emb_dataset[:] = title_embeddings
                    abstract_emb_dataset[:] = abstract_embeddings
                else:
                    # 扩展数据集
                    current_size = title_emb_dataset.shape[0]
                    new_size = current_size + len(ids)
                    title_emb_dataset.resize(new_size, axis=0)
                    abstract_emb_dataset.resize(new_size, axis=0)
                    h5_file['paper_ids'].resize(new_size, axis=0)
                    
                    # 添加新数据
                    title_emb_dataset[current_size:new_size] = title_embeddings
                    abstract_emb_dataset[current_size:new_size] = abstract_embeddings
                    h5_file['paper_ids'][current_size:new_size] = np.array(ids)
                
                # 每批次后立即同步到磁盘，确保数据不会丢失
                h5_file.flush()
            else:
                # Numpy格式，缓存嵌入向量
                title_embeddings_list.append(title_embeddings)
                abstract_embeddings_list.append(abstract_embeddings)
                numpy_batch_counter += 1
                
                # 定期将缓存的嵌入向量保存到磁盘，避免内存溢出
                if numpy_batch_counter >= numpy_save_interval or batch_idx == total_batches - 1:
                    if title_embeddings_list:
                        # 合并当前批次的嵌入向量
                        current_title_embeddings = np.vstack(title_embeddings_list)
                        current_abstract_embeddings = np.vstack(abstract_embeddings_list)
                        
                        # 批次索引标记
                        batch_suffix = f"_part{numpy_file_index}"
                        
                        # 保存为压缩numpy文件
                        current_ids = all_ids[-len(current_title_embeddings):]
                        np.savez_compressed(
                            f"{title_embedding_file}{batch_suffix}.npz", 
                            embeddings=current_title_embeddings, 
                            ids=np.array(current_ids)
                        )
                        np.savez_compressed(
                            f"{abstract_embedding_file}{batch_suffix}.npz", 
                            embeddings=current_abstract_embeddings, 
                            ids=np.array(current_ids)
                        )
                        
                        logging.info(f"已保存部分嵌入向量到批次 {numpy_file_index}")
                        
                        # 清空列表，释放内存
                        title_embeddings_list = []
                        abstract_embeddings_list = []
                        numpy_batch_counter = 0
                        numpy_file_index += 1
            
            processed_count += len(ids)
            
            # 定期输出进度信息
            if (batch_idx + 1) % save_every == 0 or batch_idx == total_batches - 1:
                # 计算处理速度和估计剩余时间
                elapsed_time = time.time() - start_time
                papers_per_second = processed_count / elapsed_time
                remaining_batches = total_batches - batch_idx - 1
                remaining_papers = remaining_batches * len(ids)  # 假设每批次大小相同
                estimated_remaining_time = remaining_papers / papers_per_second if papers_per_second > 0 else 0
                
                logging.info(f"已处理 {processed_count} 篇论文，共 {batch_idx+1}/{total_batches} 批次")
                logging.info(f"处理速度: {papers_per_second:.2f} 篇/秒")
                logging.info(f"预计剩余时间: {estimated_remaining_time/60:.2f} 分钟")
                
                # 保存当前元数据，作为备份
                with open(metadata_file + ".temp", 'w', encoding='utf-8') as f:
                    # 安全地获取模型名称
                    if use_transformers:
                        model_name = model['model'].config._name_or_path
                    else:
                        try:
                            model_name = getattr(getattr(model._first_module(), 'auto_model', None).config, '_name_or_path', '')
                        except (AttributeError, TypeError):
                            model_name = str(model.__class__.__name__)
                            
                    json.dump({
                        'papers': all_metadata,
                        'id_to_index': {paper_id: idx for idx, paper_id in enumerate(all_ids)},
                        'embedding_info': {
                            'model': model_name,
                            'embedding_dim': title_embeddings.shape[1],
                            'creation_date': datetime.now().isoformat(),
                            'use_transformers': use_transformers
                        }
                    }, f, ensure_ascii=False, indent=4)
        
        # 如果使用numpy格式，返回所有部分文件的基本名称
        if storage_format != 'h5':
            embedding_files = (title_embedding_file, abstract_embedding_file)
        else:
            embedding_files = embedding_file
        
        # 保存最终元数据
        with open(metadata_file, 'w', encoding='utf-8') as f:
            # 安全地获取模型名称
            if use_transformers:
                model_name = model['model'].config._name_or_path
            else:
                try:
                    model_name = getattr(getattr(model._first_module(), 'auto_model', None).config, '_name_or_path', '')
                except (AttributeError, TypeError):
                    model_name = str(model.__class__.__name__)
                    
            json.dump({
                'papers': all_metadata,
                'id_to_index': {paper_id: idx for idx, paper_id in enumerate(all_ids)},
                'embedding_info': {
                    'model': model_name,
                    'embedding_dim': title_embeddings.shape[1],
                    'creation_date': datetime.now().isoformat(),
                    'use_transformers': use_transformers
                }
            }, f, ensure_ascii=False, indent=2)
        
        logging.info(f"元数据已保存到: {metadata_file}")
    
    finally:
        # 关闭H5文件
        if storage_format == 'h5' and 'h5_file' in locals():
            h5_file.close()
    
    return processed_count, metadata_file, embedding_files


def validate_args(args):
    """验证命令行参数的合法性和兼容性"""
    if args.use_transformers and args.model_attn_implementation == "flash_attention_2" and not args.bf16:
        logging.warning("注意：Flash Attention 2只支持bf16和fp16精度，将自动启用bf16精度")
    
    if not args.use_transformers and args.use_flash_attention:
        logging.warning("注意：sentence-transformers不支持flash-attention，此参数在非use_transformers模式下将被忽略")
        logging.warning("如需使用flash-attention，请添加--use_transformers参数")


def main():
    parser = argparse.ArgumentParser(description="生成arXiv论文的语义嵌入")
    parser.add_argument("--input_file", type=str, default="data/arxiv/arxiv-metadata-oai-snapshot.json", 
                        help="arXiv元数据JSON文件路径")
    parser.add_argument("--output_dir", type=str, default="data/arxiv/embeddings", 
                        help="嵌入向量输出目录")
    parser.add_argument("--log_dir", type=str, default="logs", 
                        help="日志输出目录")
    parser.add_argument("--model_path", type=str, default="models/e5-mistral-7b-instruct", 
                        help="模型路径")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="批处理大小")
    parser.add_argument("--data_batch_size", type=int, default=32, 
                        help="数据加载批次大小")
    parser.add_argument("--max_seq_length", type=int, default=4096, 
                        help="最大序列长度")
    parser.add_argument("--save_every", type=int, default=10, 
                        help="每处理多少批次保存一次结果")
    parser.add_argument("--start_idx", type=int, default=0, 
                        help="从哪篇论文开始处理")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="最多处理多少篇论文，None表示处理所有")
    parser.add_argument("--start_batch", type=int, default=0, 
                        help="从哪个批次开始处理，用于断点续传")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    parser.add_argument("--storage_format", type=str, default="h5", 
                        choices=["h5", "numpy"],
                        help="嵌入向量存储格式: h5 (HDF5) 或 numpy")
    parser.add_argument("--numpy_save_interval", type=int, default=10,
                       help="当使用numpy格式时，每处理多少批次保存一次到磁盘，避免内存溢出")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="DataLoader的工作进程数")
    
    # transformers相关参数
    parser.add_argument("--use_transformers", action="store_true",
                       help="使用transformers库替代sentence-transformers，支持更多优化选项如flash-attention")
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
    
    # 验证命令行参数
    validate_args(args)
    
    # 设置日志
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(args.log_dir, log_level)
    
    # 输出运行配置
    logging.info("="*50)
    logging.info("arXiv论文摘要和标题的语义嵌入生成")
    logging.info("="*50)
    for arg, value in sorted(vars(args).items()):
        logging.info(f"参数 {arg}: {value}")
    logging.info("="*50)
    
    # 检查CUDA可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 加载模型
    logging.info(f"加载模型: {args.model_path}")
    
    # 判断是使用transformers还是sentence-transformers
    if args.use_transformers:
        model_kwargs = {}
        if torch.cuda.is_available():
            # 精度设置
            torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
            model_kwargs["torch_dtype"] = torch_dtype
            
            # 注意力机制设置
            if args.model_attn_implementation != "eager":
                model_kwargs["attn_implementation"] = args.model_attn_implementation
                if args.model_attn_implementation == "flash_attention_2":
                    logging.info("使用Flash Attention 2进行加速")
                    # Flash Attention只支持bf16或fp16，如果用户没有明确指定bf16，则在这里强制设置
                    if not args.bf16:
                        logging.warning("Flash Attention 2只支持bf16或fp16精度，自动启用bf16精度")
                        model_kwargs["torch_dtype"] = torch.bfloat16
                        args.bf16 = True  # 更新参数，以便后续日志显示正确的精度信息
        
        try:
            # 加载tokenizer和模型
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            model = AutoModel.from_pretrained(args.model_path, **model_kwargs)
            model.to(device)
            
            # 将tokenizer和model打包为一个字典，方便传递
            model_data = {
                'tokenizer': tokenizer,
                'model': model
            }
            
            precision_info = "BF16" if args.bf16 else "FP32"
            logging.info(f"已加载transformers模型，使用{precision_info}精度和{args.model_attn_implementation}注意力机制")
        except Exception as e:
            logging.error(f"加载transformers模型失败: {e}")
            raise
    else:
        try:
            # sentence-transformers不支持flash-attention，忽略use_flash_attention参数
            model = SentenceTransformer(args.model_path)
            logging.info("已加载SentenceTransformer模型，使用FP32计算")
            
            if args.use_flash_attention:
                logging.warning("注意：sentence-transformers不支持flash-attention，此参数在非use_transformers模式下将被忽略")
                logging.warning("如需使用flash-attention，请添加--use_transformers参数")
                
            model.max_seq_length = args.max_seq_length
            model.to(device)
            model_data = model  # 兼容两种API
        except Exception as e:
            logging.error(f"加载SentenceTransformer模型失败: {e}")
            raise
    
    # 加载数据集
    dataset = ArxivDataset(args.input_file, start_idx=args.start_idx, max_samples=args.max_samples)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.data_batch_size, 
        shuffle=False, 
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()  # 如果使用GPU则启用pin_memory
    )
    
    logging.info(f"数据集大小: {len(dataset)} 篇论文")
    logging.info(f"批处理大小: {args.data_batch_size}")
    logging.info(f"总批次数: {len(dataloader)}")
    logging.info(f"使用工作进程: {args.num_workers}")
    
    # 估计存储空间
    embedding_dim = 4096  # E5-Mistral-7B 的嵌入维度
    bytes_per_float = 4   # float32
    bytes_per_paper = 2 * embedding_dim * bytes_per_float  # 两个嵌入向量 (title + abstract)
    estimated_size_bytes = len(dataset) * bytes_per_paper
    estimated_size_gb = estimated_size_bytes / (1024**3)
    
    compressed_ratio = 0.25  # 假设压缩后大小约为原始大小的25%
    compressed_size_gb = estimated_size_gb * compressed_ratio
    
    logging.info(f"每篇论文的原始嵌入向量大小: {bytes_per_paper/1024:.2f} KB (使用float32)")
    logging.info(f"总共 {len(dataset)} 篇论文的原始嵌入向量估计占用空间: {estimated_size_gb:.2f} GB")
    logging.info(f"压缩后估计占用空间: {compressed_size_gb:.2f} GB (假设压缩率为 {compressed_ratio*100}%)")
    
    # 处理并保存
    start_time = time.time()
    processed_count, metadata_file, embedding_files = process_and_save(
        dataloader, 
        model_data, 
        args.output_dir, 
        batch_size=args.batch_size,
        save_every=args.save_every,
        start_batch=args.start_batch,
        storage_format=args.storage_format,
        numpy_save_interval=args.numpy_save_interval,
        use_transformers=args.use_transformers,
        task_description=args.task_description,
        max_length=args.max_seq_length
    )
    
    # 输出文件信息
    if args.storage_format == 'h5':
        embedding_file = embedding_files
        embedding_file_size = os.path.getsize(embedding_file) / (1024**2)  # MB
        logging.info(f"嵌入向量文件: {embedding_file} ({embedding_file_size:.2f} MB)")
    else:
        title_embedding_base, abstract_embedding_base = embedding_files
        logging.info(f"标题嵌入向量文件基础名: {title_embedding_base}_part*.npz")
        logging.info(f"摘要嵌入向量文件基础名: {abstract_embedding_base}_part*.npz")
        
        # 计算所有部分文件的总大小
        title_parts = list(Path(os.path.dirname(title_embedding_base)).glob(
            f"{os.path.basename(title_embedding_base)}_part*.npz"
        ))
        abstract_parts = list(Path(os.path.dirname(abstract_embedding_base)).glob(
            f"{os.path.basename(abstract_embedding_base)}_part*.npz"
        ))
        
        title_total_size = sum(os.path.getsize(f) for f in title_parts) / (1024**2)  # MB
        abstract_total_size = sum(os.path.getsize(f) for f in abstract_parts) / (1024**2)  # MB
        
        logging.info(f"标题嵌入向量文件总大小: {title_total_size:.2f} MB")
        logging.info(f"摘要嵌入向量文件总大小: {abstract_total_size:.2f} MB")
    
    metadata_file_size = os.path.getsize(metadata_file) / (1024**2)  # MB
    logging.info(f"元数据文件: {metadata_file} ({metadata_file_size:.2f} MB)")
    
    # 统计信息
    total_time = time.time() - start_time
    logging.info("="*50)
    logging.info(f"处理完成! 总共处理了 {processed_count} 篇论文")
    logging.info(f"总耗时: {total_time/60:.2f} 分钟")
    logging.info(f"平均速度: {processed_count/total_time:.2f} 篇/秒")
    logging.info("="*50)


if __name__ == "__main__":
    main()
