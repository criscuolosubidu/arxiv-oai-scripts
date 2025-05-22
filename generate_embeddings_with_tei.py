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
        # 创建可扩展的数据集
        embedding_dim = None  # 将在处理第一篇论文时确定
        title_emb_dataset = None
        abstract_emb_dataset = None
        # 创建ID数据集
        id_dtype = h5py.special_dtype(vlen=str)
        paper_ids_dataset = None
    else:  # numpy格式
        title_embedding_file = os.path.join(output_dir, f"arxiv_title_embeddings_tei_{timestamp}")
        abstract_embedding_file = os.path.join(output_dir, f"arxiv_abstract_embeddings_tei_{timestamp}")
        # 用于存储临时批次数据
        title_batch_files = []
        abstract_batch_files = []
    
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
        
        # 批次计数和总处理计数
        batch_count = 0
        total_processed = 0
        all_ids = []
        all_metadata = []
        
        # 当前批次的数据
        current_batch_ids = []
        current_batch_title_embeddings = []
        current_batch_abstract_embeddings = []
        current_batch_metadata = []
        
        # 每次最多提交这么多任务，避免内存占用过大
        max_pending_tasks = min(save_every * 3, 100000)
        
        # 函数：保存当前批次数据到磁盘并清空内存
        def save_batch_to_disk():
            nonlocal batch_count, title_emb_dataset, abstract_emb_dataset, paper_ids_dataset, embedding_dim
            
            if not current_batch_ids:
                return  # 没有数据要保存
                
            batch_title_embeddings = np.array(current_batch_title_embeddings)
            batch_abstract_embeddings = np.array(current_batch_abstract_embeddings)
            
            # 首次保存时确定嵌入维度并创建数据集
            if embedding_dim is None and len(batch_title_embeddings) > 0:
                embedding_dim = batch_title_embeddings.shape[1]
                logging.info(f"嵌入维度: {embedding_dim}")
                
                if storage_format == 'h5':
                    # 创建可扩展的数据集
                    title_emb_dataset = h5_file.create_dataset(
                        'title_embeddings', 
                        shape=(0, embedding_dim),
                        maxshape=(None, embedding_dim),
                        dtype='float32',
                        compression='gzip', 
                        compression_opts=5,
                        chunks=(min(1000, save_every), embedding_dim)
                    )
                    abstract_emb_dataset = h5_file.create_dataset(
                        'abstract_embeddings', 
                        shape=(0, embedding_dim),
                        maxshape=(None, embedding_dim),
                        dtype='float32',
                        compression='gzip', 
                        compression_opts=5,
                        chunks=(min(1000, save_every), embedding_dim)
                    )
                    # 创建ID数据集
                    paper_ids_dataset = h5_file.create_dataset(
                        'paper_ids', 
                        shape=(0,),
                        maxshape=(None,),
                        dtype=id_dtype,
                        chunks=(min(1000, save_every),)
                    )
            
            if storage_format == 'h5':
                # 扩展数据集
                current_size = title_emb_dataset.shape[0]
                new_size = current_size + len(batch_title_embeddings)
                
                title_emb_dataset.resize(new_size, axis=0)
                abstract_emb_dataset.resize(new_size, axis=0)
                paper_ids_dataset.resize(new_size, axis=0)
                
                # 将数据写入数据集
                title_emb_dataset[current_size:new_size] = batch_title_embeddings
                abstract_emb_dataset[current_size:new_size] = batch_abstract_embeddings
                paper_ids_dataset[current_size:new_size] = np.array(current_batch_ids, dtype=object)
                
                # 强制刷新到磁盘
                h5_file.flush()
            else:
                # 保存为批次numpy文件
                batch_title_file = f"{title_embedding_file}_batch_{batch_count}.npz"
                batch_abstract_file = f"{abstract_embedding_file}_batch_{batch_count}.npz"
                
                np.savez_compressed(
                    batch_title_file, 
                    embeddings=batch_title_embeddings, 
                    ids=np.array(current_batch_ids)
                )
                np.savez_compressed(
                    batch_abstract_file, 
                    embeddings=batch_abstract_embeddings, 
                    ids=np.array(current_batch_ids)
                )
                
                title_batch_files.append(batch_title_file)
                abstract_batch_files.append(batch_abstract_file)
            
            # 记录批次元数据
            all_ids.extend(current_batch_ids)
            all_metadata.extend(current_batch_metadata)
            
            # 保存临时元数据
            with open(metadata_file + f".batch_{batch_count}", 'w', encoding='utf-8') as f:
                json.dump({
                    'papers': current_batch_metadata,
                    'id_to_index': {paper_id: idx for idx, paper_id in enumerate(current_batch_ids)},
                    'batch_index': batch_count,
                    'embedding_info': {
                        'model': "TEI服务",
                        'embedding_dim': embedding_dim,
                        'creation_date': datetime.now().isoformat(),
                        'prompt_name': prompt_name
                    }
                }, f, ensure_ascii=False)
            
            # 增加批次计数
            batch_count += 1
            
            # 清空当前批次数据
            current_batch_ids.clear()
            current_batch_title_embeddings.clear()
            current_batch_abstract_embeddings.clear()
            current_batch_metadata.clear()
            
            # 输出日志
            logging.info(f"已保存批次 {batch_count}，共 {len(all_ids)} 篇论文")
        
        # 分批处理所有论文
        paper_index = 0
        while paper_index < total_papers_to_process:
            # 计算本批次要提交的任务数量
            pending_tasks = len(paper_futures) - total_processed
            tasks_to_submit = min(max_pending_tasks - pending_tasks, save_every, total_papers_to_process - paper_index)
            
            # 提交批次任务
            for _ in range(tasks_to_submit):
                paper_idx = papers_to_process[paper_index]
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
                
                paper_index += 1
            
            # 检查已完成的任务
            newly_completed_ids = []
            for paper_id in list(paper_futures.keys()):
                if (paper_id not in newly_completed_ids and 
                    title_embedding_futures[paper_id].done() and
                    abstract_embedding_futures[paper_id].done()):
                    try:
                        # 获取嵌入结果
                        title_embedding = title_embedding_futures[paper_id].result()
                        abstract_embedding = abstract_embedding_futures[paper_id].result()
                        
                        # 存储到当前批次
                        current_batch_ids.append(paper_id)
                        current_batch_title_embeddings.append(title_embedding)
                        current_batch_abstract_embeddings.append(abstract_embedding)
                        
                        paper = paper_futures[paper_id]
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
                        current_batch_metadata.append(paper_metadata)
                        
                        newly_completed_ids.append(paper_id)
                        total_processed += 1
                        
                        # 清理已完成的任务
                        del paper_futures[paper_id]
                        del title_embedding_futures[paper_id]
                        del abstract_embedding_futures[paper_id]
                    except Exception as e:
                        logging.error(f"处理论文 {paper_id} 时出错: {str(e)}")
                        # 标记为已完成，但失败
                        newly_completed_ids.append(paper_id)
                        total_processed += 1
                        
                        # 清理已完成的任务
                        del paper_futures[paper_id]
                        del title_embedding_futures[paper_id]
                        del abstract_embedding_futures[paper_id]
            
            # 更新进度条
            pbar.update(len(newly_completed_ids))
            
            # 当前批次达到保存阈值，刷新到磁盘
            if len(current_batch_ids) >= save_every:
                # 计算处理速度
                elapsed_time = time.time() - start_time
                papers_per_second = total_processed / elapsed_time if elapsed_time > 0 else 0
                estimated_remaining_time = (total_papers_to_process - total_processed) / papers_per_second if papers_per_second > 0 else 0
                
                logging.info(f"已处理 {total_processed}/{total_papers_to_process} 篇论文")
                logging.info(f"处理速度: {papers_per_second:.2f} 篇/秒")
                logging.info(f"预计剩余时间: {estimated_remaining_time/60:.2f} 分钟")
                
                # 保存当前批次到磁盘
                save_batch_to_disk()
            
            # 短暂休眠，避免CPU占用过高
            time.sleep(0.1)
        
        # 等待所有剩余任务完成
        while paper_futures:
            newly_completed_ids = []
            for paper_id in list(paper_futures.keys()):
                if (paper_id not in newly_completed_ids and 
                    title_embedding_futures[paper_id].done() and
                    abstract_embedding_futures[paper_id].done()):
                    try:
                        # 获取嵌入结果
                        title_embedding = title_embedding_futures[paper_id].result()
                        abstract_embedding = abstract_embedding_futures[paper_id].result()
                        
                        # 存储到当前批次
                        current_batch_ids.append(paper_id)
                        current_batch_title_embeddings.append(title_embedding)
                        current_batch_abstract_embeddings.append(abstract_embedding)
                        
                        paper = paper_futures[paper_id]
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
                        current_batch_metadata.append(paper_metadata)
                        
                        newly_completed_ids.append(paper_id)
                        total_processed += 1
                        
                        # 清理已完成的任务
                        del paper_futures[paper_id]
                        del title_embedding_futures[paper_id]
                        del abstract_embedding_futures[paper_id]
                    except Exception as e:
                        logging.error(f"处理论文 {paper_id} 时出错: {str(e)}")
                        newly_completed_ids.append(paper_id)
                        total_processed += 1
                        
                        # 清理已完成的任务
                        del paper_futures[paper_id]
                        del title_embedding_futures[paper_id]
                        del abstract_embedding_futures[paper_id]
            
            # 更新进度条
            pbar.update(len(newly_completed_ids))
            
            # 如果有已完成的任务，检查是否需要保存
            if newly_completed_ids and len(current_batch_ids) >= save_every:
                save_batch_to_disk()
                
            # 如果没有新完成的任务，等待一会儿
            if not newly_completed_ids:
                time.sleep(0.5)
        
        # 保存最后一个批次（如果有）
        if current_batch_ids:
            save_batch_to_disk()
            
        pbar.close()
    
    # 检查是否有嵌入向量
    if not all_ids:
        logging.error("没有成功生成任何嵌入向量")
        if storage_format == 'h5':
            h5_file.close()
        return 0, metadata_file, None
    
    # 如果使用numpy格式，将所有批次合并为最终文件
    if storage_format != 'h5':
        logging.info("合并所有批次文件...")
        
        # 合并标题嵌入
        final_title_file = f"{title_embedding_file}.npz"
        # 合并摘要嵌入
        final_abstract_file = f"{abstract_embedding_file}.npz"
        
        # 保存ID索引到最终文件
        np.savez_compressed(final_title_file + ".ids", ids=np.array(all_ids))
        np.savez_compressed(final_abstract_file + ".ids", ids=np.array(all_ids))
        
        # 删除临时批次文件
        for batch_file in title_batch_files:
            try:
                os.remove(batch_file)
            except Exception as e:
                logging.warning(f"删除临时文件 {batch_file} 时出错: {str(e)}")
                
        for batch_file in abstract_batch_files:
            try:
                os.remove(batch_file)
            except Exception as e:
                logging.warning(f"删除临时文件 {batch_file} 时出错: {str(e)}")
                
        embedding_files = (final_title_file, final_abstract_file)
    else:
        # HDF5格式已经写入文件，关闭即可
        h5_file.close()
        embedding_files = embedding_file
    
    # 保存最终元数据
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            'papers': all_metadata,
            'id_to_index': {paper_id: idx for idx, paper_id in enumerate(all_ids)},
            'embedding_info': {
                'model': "TEI服务",
                'embedding_dim': embedding_dim,
                'creation_date': datetime.now().isoformat(),
                'prompt_name': prompt_name,
                'total_papers': len(all_ids),
                'batch_count': batch_count
            }
        }, f, ensure_ascii=False, indent=2)
    
    # 删除临时元数据文件
    for i in range(batch_count):
        temp_metadata_file = metadata_file + f".batch_{i}"
        try:
            if os.path.exists(temp_metadata_file):
                os.remove(temp_metadata_file)
        except Exception as e:
            logging.warning(f"删除临时元数据文件 {temp_metadata_file} 时出错: {str(e)}")
    
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
    parser.add_argument("--save_every", type=int, default=10000, 
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
