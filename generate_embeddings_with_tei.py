#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用Text Embedding Inference (TEI)服务生成arXiv论文的语义嵌入
通过HTTP API调用本地TEI服务，对arXiv论文的标题和摘要生成嵌入向量
"""

import json
import os
import time
import argparse
import numpy as np
import logging
import h5py
import requests
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import concurrent.futures


# 配置日志
def setup_logger(log_dir, log_level=logging.INFO):
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名，包含时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"tei_embedding_generation_{timestamp}.log")
    
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


class ArxivDataset:
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
        # tokenizer error for e5-mistral-7b-instruct
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


def generate_embeddings_batch_parallel(texts, tei_url, prompt_name=None, max_workers=4):
    """
    并行批量生成嵌入向量
    
    参数:
        texts: 文本列表
        tei_url: TEI服务的URL
        prompt_name: 可选的提示名称
        max_workers: 并行工作线程数
        
    返回:
        numpy数组形式的嵌入向量
    """
    
    # 使用线程池并行处理请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_text = {
            executor.submit(call_tei_service, text, tei_url, prompt_name): i 
            for i, text in enumerate(texts)
        }
        
        # 按原始顺序收集结果
        results = [None] * len(texts)
        for future in concurrent.futures.as_completed(future_to_text):
            idx = future_to_text[future]
            try:
                embedding = future.result()
                results[idx] = embedding
            except Exception as e:
                logging.error(f"处理文本 {idx} 时出错: {str(e)}")
                raise
    
    # 合并结果
    return np.array(results)


def process_and_save(
        dataset, 
        tei_url,
        output_dir, 
        max_workers=16,
        save_every=1000, 
        start_idx=0, 
        storage_format='h5', 
        prompt_name=None
    ):
    """处理数据并保存嵌入向量和元数据 - 完全并行处理版本"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确定保存格式
    if storage_format == 'h5':
        embedding_file = os.path.join(output_dir, f"arxiv_embeddings_tei_{timestamp}.h5")
        h5_file = h5py.File(embedding_file, 'w')
    else:  # numpy格式
        title_embedding_file = os.path.join(output_dir, f"arxiv_title_embeddings_tei_{timestamp}.npz")
        abstract_embedding_file = os.path.join(output_dir, f"arxiv_abstract_embeddings_tei_{timestamp}.npz")
    
    # 元数据文件
    metadata_file = os.path.join(output_dir, f"arxiv_metadata_tei_{timestamp}.json")
    
    logging.info(f"将向量保存到: {output_dir}")
    logging.info(f"将元数据保存到: {metadata_file}")
    
    start_time = time.time()
    total_papers = len(dataset)
    
    # 跳过已处理的论文
    papers_to_process = range(start_idx, total_papers)
    total_papers_to_process = len(papers_to_process)
    
    # 使用并发执行器处理所有论文
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建进度条
        pbar = tqdm(total=total_papers_to_process, desc="处理论文")
        
        # 用于存储所有任务的future对象
        paper_futures = {}
        title_embedding_futures = {}
        abstract_embedding_futures = {}
        
        # 提交所有论文的标题和摘要嵌入任务
        for paper_idx in papers_to_process:
            paper = dataset[paper_idx]
            paper_id = paper['id']
            title = paper['title']
            abstract = paper['abstract']
            
            # 提交嵌入任务
            title_future = executor.submit(call_tei_service, title, tei_url, prompt_name)
            abstract_future = executor.submit(call_tei_service, abstract, tei_url, prompt_name)
            
            # 存储future对象
            paper_futures[paper_id] = paper
            title_embedding_futures[paper_id] = title_future
            abstract_embedding_futures[paper_id] = abstract_future
        
        # 定期检查完成的任务并更新进度
        completed_count = 0
        processed_count = 0
        title_embeddings_dict = {}
        abstract_embeddings_dict = {}
        all_ids = []
        all_metadata = []
        
        # 循环检查任务完成情况
        while completed_count < total_papers_to_process:
            # 获取新完成的任务
            newly_completed = []
            
            for paper_id in paper_futures:
                if (paper_id not in title_embeddings_dict and 
                    title_embedding_futures[paper_id].done() and
                    abstract_embedding_futures[paper_id].done()):
                    try:
                        # 获取嵌入结果
                        title_embedding = title_embedding_futures[paper_id].result()
                        abstract_embedding = abstract_embedding_futures[paper_id].result()
                        
                        # 存储嵌入和元数据
                        title_embeddings_dict[paper_id] = title_embedding
                        abstract_embeddings_dict[paper_id] = abstract_embedding
                        
                        paper = paper_futures[paper_id]
                        all_ids.append(paper_id)
                        paper_metadata = {
                            'id': paper_id,
                            'title': paper['title'],
                            'abstract': paper['abstract'],
                            'authors': paper['authors'],
                            'categories': paper['categories'],
                            'journal_ref': paper['journal_ref'],
                            'doi': paper['doi'],
                            'update_date': paper['update_date']
                        }
                        all_metadata.append(paper_metadata)
                        
                        newly_completed.append(paper_id)
                        completed_count += 1
                    except Exception as e:
                        logging.error(f"处理论文 {paper_id} 时出错: {str(e)}")
                        # 标记为已完成，但失败
                        newly_completed.append(paper_id)
                        completed_count += 1
            
            # 更新进度条
            pbar.update(len(newly_completed))
            processed_count += len(newly_completed)
            
            # 定期保存进度
            if processed_count >= save_every or completed_count >= total_papers_to_process:
                # 计算处理速度
                elapsed_time = time.time() - start_time
                papers_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
                estimated_remaining_time = (total_papers_to_process - completed_count) / papers_per_second if papers_per_second > 0 else 0
                
                logging.info(f"已处理 {processed_count}/{total_papers_to_process} 篇论文")
                logging.info(f"处理速度: {papers_per_second:.2f} 篇/秒")
                logging.info(f"预计剩余时间: {estimated_remaining_time/60:.2f} 分钟")
                
                # 临时保存元数据
                with open(metadata_file + ".temp", 'w', encoding='utf-8') as f:
                    json.dump({
                        'papers': all_metadata,
                        'id_to_index': {paper_id: idx for idx, paper_id in enumerate(all_ids)},
                        'embedding_info': {
                            'model': "TEI服务",
                            'embedding_dim': next(iter(title_embeddings_dict.values())).shape[0] if title_embeddings_dict else 0,
                            'creation_date': datetime.now().isoformat(),
                            'prompt_name': prompt_name
                        }
                    }, f, ensure_ascii=False, indent=4)
                
                processed_count = 0
            
            # 短暂休眠，避免CPU占用过高
            time.sleep(0.1)
        
        pbar.close()
    
    # 检查是否有嵌入向量
    if not title_embeddings_dict:
        logging.error("没有成功生成任何嵌入向量")
        return 0, metadata_file, None
    
    # 根据all_ids的顺序排列嵌入向量
    title_embeddings = np.array([title_embeddings_dict[paper_id] for paper_id in all_ids])
    abstract_embeddings = np.array([abstract_embeddings_dict[paper_id] for paper_id in all_ids])
    
    # 保存嵌入向量
    if storage_format == 'h5':
        embedding_dim = title_embeddings.shape[1]
        # 创建数据集
        title_emb_dataset = h5_file.create_dataset(
            'title_embeddings', 
            data=title_embeddings,
            dtype='float32',
            compression='gzip', 
            compression_opts=5
        )
        abstract_emb_dataset = h5_file.create_dataset(
            'abstract_embeddings', 
            data=abstract_embeddings,
            dtype='float32',
            compression='gzip', 
            compression_opts=5
        )
        # 存储ID索引
        id_dtype = h5py.special_dtype(vlen=str)
        h5_file.create_dataset(
            'paper_ids', 
            data=np.array(all_ids, dtype=object), 
            dtype=id_dtype, 
            chunks=True
        )
        
        h5_file.close()
        embedding_files = embedding_file
    else:
        # 保存为numpy格式
        np.savez_compressed(
            title_embedding_file, 
            embeddings=title_embeddings, 
            ids=np.array(all_ids)
        )
        np.savez_compressed(
            abstract_embedding_file, 
            embeddings=abstract_embeddings, 
            ids=np.array(all_ids)
        )
        
        embedding_files = (title_embedding_file, abstract_embedding_file)
    
    # 保存最终元数据
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            'papers': all_metadata,
            'id_to_index': {paper_id: idx for idx, paper_id in enumerate(all_ids)},
            'embedding_info': {
                'model': "TEI服务",
                'embedding_dim': title_embeddings.shape[1],
                'creation_date': datetime.now().isoformat(),
                'prompt_name': prompt_name
            }
        }, f, ensure_ascii=False, indent=2)
    
    logging.info(f"元数据已保存到: {metadata_file}")
    
    return len(all_ids), metadata_file, embedding_files


def main():
    parser = argparse.ArgumentParser(description="使用TEI服务生成arXiv论文的语义嵌入")
    parser.add_argument("--input_file", type=str, default="data/arxiv/arxiv-metadata-oai-snapshot.json", 
                        help="arXiv元数据JSON文件路径")
    parser.add_argument("--output_dir", type=str, default="data/arxiv/embeddings", 
                        help="嵌入向量输出目录")
    parser.add_argument("--log_dir", type=str, default="logs", 
                        help="日志输出目录")
    parser.add_argument("--tei_url", type=str, default="http://127.0.0.1:8080/embed", 
                        help="TEI服务URL")
    parser.add_argument("--max_workers", type=int, default=16, 
                        help="并行工作线程最大数量")
    parser.add_argument("--save_every", type=int, default=1000, 
                        help="每处理多少篇论文保存一次进度")
    parser.add_argument("--start_idx", type=int, default=0, 
                        help="从哪篇论文开始处理")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="最多处理多少篇论文，None表示处理所有")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    parser.add_argument("--storage_format", type=str, default="h5", 
                        choices=["h5", "numpy"],
                        help="嵌入向量存储格式: h5 (HDF5) 或 numpy")
    parser.add_argument("--prompt_name", type=str, default=None,
                       help="TEI服务的提示名称")
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(args.log_dir, log_level)
    
    # 输出运行配置
    logging.info("="*50)
    logging.info("使用TEI服务生成arXiv论文摘要和标题的语义嵌入 - 完全并行版本")
    logging.info("="*50)
    for arg, value in sorted(vars(args).items()):
        logging.info(f"参数 {arg}: {value}")
    logging.info("="*50)
    
    # 测试TEI服务连接
    test_text = "测试TEI服务连接"
    try:
        test_embedding = call_tei_service(test_text, args.tei_url, args.prompt_name)
        embedding_dim = test_embedding.shape[0]
        logging.info(f"TEI服务连接成功，嵌入维度: {embedding_dim}")
    except Exception as e:
        logging.error(f"TEI服务连接测试失败: {str(e)}")
        logging.error("请确保TEI服务正在运行，并且可以通过指定的URL访问")
        return
    
    # 加载数据集
    dataset = ArxivDataset(args.input_file, start_idx=0, max_samples=args.max_samples)
    
    logging.info(f"数据集大小: {len(dataset)} 篇论文")
    logging.info(f"将从索引 {args.start_idx} 开始处理")
    logging.info(f"并行工作线程数: {args.max_workers}")
    
    # 估计存储空间
    bytes_per_float = 4   # float32
    bytes_per_paper = 2 * embedding_dim * bytes_per_float  # 两个嵌入向量 (title + abstract)
    paper_count = len(dataset) - args.start_idx
    if args.max_samples and args.max_samples < paper_count:
        paper_count = args.max_samples
    estimated_size_bytes = paper_count * bytes_per_paper
    estimated_size_gb = estimated_size_bytes / (1024**3)
    
    compressed_ratio = 0.25  # 假设压缩后大小约为原始大小的25%
    compressed_size_gb = estimated_size_gb * compressed_ratio
    
    logging.info(f"每篇论文的原始嵌入向量大小: {bytes_per_paper/1024:.2f} KB (使用float32)")
    logging.info(f"总共 {paper_count} 篇论文的原始嵌入向量估计占用空间: {estimated_size_gb:.2f} GB")
    logging.info(f"压缩后估计占用空间: {compressed_size_gb:.2f} GB (假设压缩率为 {compressed_ratio*100}%)")
    
    # 处理并保存
    start_time = time.time()
    processed_count, metadata_file, embedding_files = process_and_save(
        dataset, 
        args.tei_url, 
        args.output_dir, 
        max_workers=args.max_workers,
        save_every=args.save_every,
        start_idx=args.start_idx,
        storage_format=args.storage_format,
        prompt_name=args.prompt_name
    )
    
    # 输出文件信息
    if args.storage_format == 'h5':
        embedding_file = embedding_files
        if embedding_file and os.path.exists(embedding_file):
            embedding_file_size = os.path.getsize(embedding_file) / (1024**2)  # MB
            logging.info(f"嵌入向量文件: {embedding_file} ({embedding_file_size:.2f} MB)")
    else:
        title_embedding_file, abstract_embedding_file = embedding_files
        if title_embedding_file and os.path.exists(title_embedding_file):
            title_file_size = os.path.getsize(title_embedding_file) / (1024**2)  # MB
            abstract_file_size = os.path.getsize(abstract_embedding_file) / (1024**2)  # MB
            logging.info(f"标题嵌入向量文件: {title_embedding_file} ({title_file_size:.2f} MB)")
            logging.info(f"摘要嵌入向量文件: {abstract_embedding_file} ({abstract_file_size:.2f} MB)")
    
    if metadata_file and os.path.exists(metadata_file):
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
