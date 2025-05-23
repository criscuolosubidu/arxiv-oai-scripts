#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用TEI服务生成arXiv论文的语义嵌入 - 内存优化版本V3
平衡GPU吞吐量与内存管理：优化批次大小以充分利用40t/s的TEI性能
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
from tqdm import tqdm
import gc
import psutil
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading


def setup_logger(log_dir, log_level=logging.INFO):
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"tei_embedding_generation_v3_{timestamp}.log")
    
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(file_format)
    logger.addHandler(console_handler)
    
    return logger


def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB


class StreamingArxivDataset:
    """流式arXiv数据集，逐行读取，不预加载"""
    
    def __init__(self, file_path, start_idx=0, max_samples=None):
        self.file_path = file_path
        self.start_idx = start_idx
        self.max_samples = max_samples
        
    def stream_papers(self):
        """流式迭代论文数据"""
        count = 0
        processed = 0
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # 跳过前面的行
            for _ in range(self.start_idx):
                next(f, None)
                count += 1
            
            for line in f:
                try:
                    paper = json.loads(line)
                    count += 1
                    
                    # 检查必要字段
                    if 'title' in paper and 'abstract' in paper and paper['title'] and paper['abstract']:
                        yield {
                            'id': paper.get('id', ''),
                            'title': paper.get('title', '').replace('\n', ' '),
                            'abstract': paper.get('abstract', '').replace('\n', ' '),
                            'authors': paper.get('authors', ''),
                            'categories': paper.get('categories', ''),
                            'journal_ref': paper.get('journal-ref', ''),
                            'doi': paper.get('doi', ''),
                            'update_date': paper.get('update_date', '')
                        }
                        processed += 1
                        
                        if self.max_samples and processed >= self.max_samples:
                            break
                            
                except json.JSONDecodeError:
                    continue


async def call_tei_service_async(session, text, tei_url, prompt_name=None, max_retries=3):
    """异步调用TEI服务获取文本嵌入向量"""
    payload = {"inputs": text}
    if prompt_name:
        payload['inputs'] = prompt_name + " " + text

    headers = {"Content-Type": "application/json"}
    
    for attempt in range(max_retries):
        try:
            async with session.post(tei_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    embedding = np.array(result[0], dtype=np.float16)
                    return embedding
                elif response.status == 429:
                    await asyncio.sleep(1 * (2 ** attempt))
                    continue
                else:
                    logging.error(f"TEI服务错误，状态码: {response.status}")
                    raise ValueError(f"TEI服务返回错误，状态码: {response.status}")
                    
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (2 ** attempt))
            else:
                raise ValueError(f"调用TEI服务失败: {str(e)}")
    
    raise ValueError(f"达到最大重试次数 ({max_retries})")


def call_tei_service_sync(text, tei_url, prompt_name=None, max_retries=3, retry_delay=1):
    """同步调用TEI服务获取文本嵌入向量"""
    payload = {"inputs": text, "normalize": False}  # 默认不使用normalize，方便后续transformers的兼容
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
                time.sleep(retry_delay * (2 ** attempt))
                continue
            else:
                logging.error(f"TEI服务错误，状态码: {response.status_code}")
                raise ValueError(f"TEI服务返回错误，状态码: {response.status_code}")
                
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
            else:
                raise ValueError(f"调用TEI服务失败: {str(e)}")
    
    raise ValueError(f"达到最大重试次数 ({max_retries})")


class HDF5Writer:
    """线程安全的HDF5文件写入器，支持增量写入和压缩"""
    
    def __init__(self, output_file, embedding_dim):
        self.output_file = output_file
        self.embedding_dim = embedding_dim
        self.h5_file = h5py.File(output_file, 'w')
        self.lock = threading.Lock()
        
        # 创建可扩展的数据集
        self.title_dataset = self.h5_file.create_dataset(
            'title_embeddings',
            shape=(0, embedding_dim),
            maxshape=(None, embedding_dim),
            dtype=np.float16,
            compression='gzip',
            compression_opts=9,
            chunks=(1000, embedding_dim)
        )
        
        self.abstract_dataset = self.h5_file.create_dataset(
            'abstract_embeddings',
            shape=(0, embedding_dim),
            maxshape=(None, embedding_dim),
            dtype=np.float16,
            compression='gzip',
            compression_opts=9,
            chunks=(1000, embedding_dim)
        )
        
        # ID数据集
        id_dtype = h5py.special_dtype(vlen=str)
        self.ids_dataset = self.h5_file.create_dataset(
            'paper_ids',
            shape=(0,),
            maxshape=(None,),
            dtype=id_dtype,
            chunks=(1000,)
        )
        
        self.current_size = 0
        
    def append_batch(self, ids, title_embeddings, abstract_embeddings):
        """线程安全地追加一批数据到HDF5文件"""
        with self.lock:
            batch_size = len(ids)
            if batch_size == 0:
                return
                
            # 扩展数据集
            new_size = self.current_size + batch_size
            self.title_dataset.resize(new_size, axis=0)
            self.abstract_dataset.resize(new_size, axis=0)
            self.ids_dataset.resize(new_size, axis=0)
            
            # 写入数据
            title_array = np.array(title_embeddings, dtype=np.float16)
            abstract_array = np.array(abstract_embeddings, dtype=np.float16)
            
            self.title_dataset[self.current_size:new_size] = title_array
            self.abstract_dataset[self.current_size:new_size] = abstract_array
            self.ids_dataset[self.current_size:new_size] = np.array(ids, dtype=object)
            
            # 立即刷新到磁盘
            self.h5_file.flush()
            self.current_size = new_size
            
            # 删除临时数组，释放内存
            del title_array, abstract_array
        
    def close(self):
        """关闭HDF5文件"""
        with self.lock:
            self.h5_file.close()


async def process_batch_async(batch_papers, tei_url, prompt_name=None, max_concurrent=10):
    """异步批次处理，充分利用GPU并发能力"""
    
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=60),
        connector=aiohttp.TCPConnector(limit=max_concurrent)
    ) as session:
        
        # 创建所有任务，保持与原始论文的对应关系
        tasks = []
        for i, paper in enumerate(batch_papers):
            # 为每篇论文创建标题和摘要的嵌入任务
            title_task = call_tei_service_async(session, paper['title'], tei_url, prompt_name)
            abstract_task = call_tei_service_async(session, paper['abstract'], tei_url, prompt_name)
            tasks.append((i, paper, title_task, abstract_task))
        
        # 收集结果，保证一一对应
        results = [None] * len(batch_papers)  # 预分配结果数组
        
        for i, paper, title_task, abstract_task in tasks:
            try:
                title_embedding = await title_task
                abstract_embedding = await abstract_task
                results[i] = (paper['id'], title_embedding, abstract_embedding)
            except Exception as e:
                logging.error(f"处理论文 {paper['id']} 时出错: {str(e)}")
                results[i] = None  # 失败的论文用None占位
        
        # 过滤掉失败的结果，但保持成功结果的一一对应
        batch_ids = []
        batch_title_embeddings = []
        batch_abstract_embeddings = []
        
        for result in results:
            if result is not None:
                paper_id, title_embedding, abstract_embedding = result
                batch_ids.append(paper_id)
                batch_title_embeddings.append(title_embedding)
                batch_abstract_embeddings.append(abstract_embedding)
        
        return batch_ids, batch_title_embeddings, batch_abstract_embeddings


def process_batch_sync(batch_papers, tei_url, prompt_name=None, max_workers=10):
    """同步批次处理，使用线程池并发"""
    
    def process_single_paper(paper):
        try:
            title_embedding = call_tei_service_sync(paper['title'], tei_url, prompt_name)
            abstract_embedding = call_tei_service_sync(paper['abstract'], tei_url, prompt_name)
            return paper['id'], title_embedding, abstract_embedding
        except Exception as e:
            logging.error(f"处理论文 {paper['id']} 时出错: {str(e)}")
            return None
    
    # 使用线程池处理，保持顺序
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_paper, batch_papers))
    
    # 过滤掉失败的结果，但保持成功结果的一一对应
    batch_ids = []
    batch_title_embeddings = []
    batch_abstract_embeddings = []
    
    for result in results:
        if result is not None:
            paper_id, title_embedding, abstract_embedding = result
            batch_ids.append(paper_id)
            batch_title_embeddings.append(title_embedding)
            batch_abstract_embeddings.append(abstract_embedding)
    
    return batch_ids, batch_title_embeddings, batch_abstract_embeddings


async def process_and_save(
    dataset, 
    tei_url,
    output_dir, 
    batch_size=80,          
    save_every=1000,        
    prompt_name=None,
    use_async=True,
    max_concurrent=20
):
    """流式处理"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    embedding_file = os.path.join(output_dir, f"arxiv_embeddings_{timestamp}.h5")
    metadata_file = os.path.join(output_dir, f"arxiv_metadata_{timestamp}.json")
    
    logging.info(f"流式处理")
    logging.info(f"批处理大小: {batch_size}")
    logging.info(f"每 {save_every} 篇论文保存一次到磁盘")
    logging.info(f"使用{'异步' if use_async else '同步'}处理")
    logging.info(f"最大并发数: {max_concurrent}")
    
    # 首先获取嵌入维度
    test_embedding = call_tei_service_sync("TEST TEXT", tei_url, prompt_name)
    embedding_dim = len(test_embedding)
    logging.info(f"嵌入维度: {embedding_dim}")
    
    # 初始化HDF5写入器
    hdf5_writer = HDF5Writer(embedding_file, embedding_dim)
    
    start_time = time.time()
    total_processed = 0
    batch_papers = []
    
    # 用于统计吞吐量
    last_time = start_time
    last_processed = 0
    
    # 创建进度条
    pbar = tqdm(desc="处理论文", unit="篇")
    
    try:
        async for paper in async_stream_papers(dataset):
            batch_papers.append(paper)
            
            # 当批次达到指定大小时处理
            if len(batch_papers) >= batch_size:
                batch_start_time = time.time()
                
                # 处理当前批次
                if use_async:
                    batch_ids, batch_title_embeddings, batch_abstract_embeddings = await process_batch_async(
                        batch_papers, tei_url, prompt_name, max_concurrent
                    )
                else:
                    batch_ids, batch_title_embeddings, batch_abstract_embeddings = process_batch_sync(
                        batch_papers, tei_url, prompt_name, max_concurrent
                    )
                
                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                batch_throughput = len(batch_ids) / batch_duration if batch_duration > 0 else 0
                
                if batch_ids:
                    total_processed += len(batch_ids)
                    
                    # 立即写入HDF5文件
                    hdf5_writer.append_batch(batch_ids, batch_title_embeddings, batch_abstract_embeddings)
                    
                    # 立即释放内存
                    del batch_ids, batch_title_embeddings, batch_abstract_embeddings
                    gc.collect()
                    
                    # 计算当前吞吐量
                    current_time = time.time()
                    if current_time - last_time >= 10:  # 每10秒计算一次吞吐量
                        recent_throughput = (total_processed - last_processed) / (current_time - last_time)
                        memory_mb = get_memory_usage()
                        logging.info(f"已处理 {total_processed} 篇论文 | "
                                   f"批次吞吐量: {batch_throughput:.1f} 篇/秒 | "
                                   f"平均吞吐量: {recent_throughput:.1f} 篇/秒 | "
                                   f"内存使用: {memory_mb:.2f} MB")
                        last_time = current_time
                        last_processed = total_processed
                
                # 清空批次
                batch_papers.clear()
                pbar.update(batch_size)
                
                # 避免过快请求
                await asyncio.sleep(0.01)
        
        # 处理最后剩余的论文
        if batch_papers:
            if use_async:
                batch_ids, batch_title_embeddings, batch_abstract_embeddings = await process_batch_async(
                    batch_papers, tei_url, prompt_name, max_concurrent
                )
            else:
                batch_ids, batch_title_embeddings, batch_abstract_embeddings = process_batch_sync(
                    batch_papers, tei_url, prompt_name, max_concurrent
                )
            
            if batch_ids:
                hdf5_writer.append_batch(batch_ids, batch_title_embeddings, batch_abstract_embeddings)
                total_processed += len(batch_ids)
                del batch_ids, batch_title_embeddings, batch_abstract_embeddings
        
        pbar.close()
        
        # 关闭HDF5文件
        hdf5_writer.close()
        
        # 保存元数据
        final_metadata = {
            'embedding_info': {
                'model': "TEI服务 (E5-Mistral-7B)",
                'embedding_dim': embedding_dim,
                'creation_date': datetime.now().isoformat(),
                'prompt_name': prompt_name,
                'total_papers': total_processed,
                'processing_method': 'streaming_gpu_optimized_v3',
                'data_type': 'float16',
                'compression': 'gzip_level_9',
                'batch_size': batch_size,
                'max_concurrent': max_concurrent,
                'async_processing': use_async
            }
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(final_metadata, f, ensure_ascii=False, indent=2)
        
        return total_processed, metadata_file, embedding_file
        
    except Exception as e:
        logging.error(f"处理过程中出错: {str(e)}")
        hdf5_writer.close()
        raise


async def async_stream_papers(dataset):
    """将同步迭代器转换为异步迭代器"""
    for paper in dataset.stream_papers():
        yield paper
        await asyncio.sleep(0)  # 让出控制权


def main():
    parser = argparse.ArgumentParser(description="GPU吞吐量优化的arXiv嵌入向量生成 - V3")
    parser.add_argument("--input_file", type=str, default="data/arxiv/arxiv-metadata-oai-snapshot.json", 
                        help="arXiv元数据JSON文件路径")
    parser.add_argument("--output_dir", type=str, default="data/arxiv/embeddings", 
                        help="嵌入向量输出目录")
    parser.add_argument("--log_dir", type=str, default="logs", 
                        help="日志输出目录")
    parser.add_argument("--tei_url", type=str, default="http://127.0.0.1:8080/embed", 
                        help="TEI服务URL")
    parser.add_argument("--batch_size", type=int, default=80, 
                        help="批处理大小（优化GPU利用率，推荐50-100）")
    parser.add_argument("--save_every", type=int, default=1000, 
                        help="每处理多少篇论文保存一次进度")
    parser.add_argument("--max_concurrent", type=int, default=20, 
                        help="最大并发请求数")
    parser.add_argument("--start_idx", type=int, default=0, 
                        help="从哪篇论文开始处理")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="最多处理多少篇论文，None表示处理所有")
    parser.add_argument("--use_async", action="store_true", default=True,
                        help="使用异步处理（推荐）")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    parser.add_argument("--prompt_name", type=str, default=None,
                       help="TEI服务的提示名称")
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(args.log_dir, log_level)
    
    # 输出运行配置
    logging.info("="*60)
    logging.info("使用TEI进行arXiv论文标题和摘要的embedding生成")
    logging.info("="*60)
    for arg, value in sorted(vars(args).items()):
        logging.info(f"参数 {arg}: {value}")
    logging.info("="*60)
    
    logging.info(f"- 当前批次大小: {args.batch_size} 篇")
    logging.info(f"- 预期批次处理时间: {args.batch_size / 40:.1f} 秒")
    logging.info(f"- 最大并发请求: {args.max_concurrent}")
    batch_memory_mb = args.batch_size * 32 / 1024  # 32KB per paper
    logging.info(f"- 单批次内存占用: {batch_memory_mb:.1f} MB")
    logging.info(f"- 保存间隔内存峰值: {args.save_every * 32 / 1024:.1f} MB")
    
    # 测试TEI服务连接
    test_text = "This is a test text for TEI service connection."
    try:
        test_embedding = call_tei_service_sync(test_text, args.tei_url, args.prompt_name)
        embedding_dim = test_embedding.shape[0]
        actual_size_per_paper = embedding_dim * 2 * 2  # 2个向量 * 2字节(fp16)
        logging.info(f"TEI服务连接成功，嵌入维度: {embedding_dim}")
        logging.info(f"实际每篇论文嵌入向量大小: {actual_size_per_paper / 1024:.1f}KB")
        
        # 重新计算内存使用
        actual_batch_memory = args.batch_size * actual_size_per_paper / 1024 / 1024
        logging.info(f"实际单批次内存占用: {actual_batch_memory:.1f} MB")
        
    except Exception as e:
        logging.error(f"TEI服务连接测试失败: {str(e)}")
        return
    
    # 创建流式数据集
    dataset = StreamingArxivDataset(args.input_file, start_idx=args.start_idx, max_samples=args.max_samples)
    
    # 输出初始内存使用情况
    initial_memory = get_memory_usage()
    logging.info(f"初始内存使用: {initial_memory:.2f} MB")
    
    # 异步处理并保存
    async def run_processing():
        start_time = time.time()
        processed_count, metadata_file, embedding_file = await process_and_save(
            dataset, 
            args.tei_url, 
            args.output_dir, 
            batch_size=args.batch_size,
            save_every=args.save_every,
            prompt_name=args.prompt_name,
            use_async=args.use_async,
            max_concurrent=args.max_concurrent
        )
        
        # 输出最终统计
        total_time = time.time() - start_time
        final_memory = get_memory_usage()
        average_throughput = processed_count / total_time
        
        # 检查输出文件大小
        if os.path.exists(embedding_file):
            file_size_gb = os.path.getsize(embedding_file) / (1024**3)
            logging.info(f"HDF5文件大小: {file_size_gb:.2f} GB")
            compression_ratio = file_size_gb / (processed_count * actual_size_per_paper / (1024**3))
            logging.info(f"压缩率: {compression_ratio:.2f}")
        
        logging.info("="*60)
        logging.info(f"处理完成! 总共处理了 {processed_count} 篇论文")
        logging.info(f"总耗时: {total_time/60:.2f} 分钟")
        logging.info(f"平均吞吐量: {average_throughput:.2f} 篇/秒")
        logging.info(f"GPU利用率: {average_throughput/40*100:.1f}% (目标:接近100%)")
        logging.info(f"最终内存使用: {final_memory:.2f} MB")
        logging.info(f"内存增长: {final_memory - initial_memory:.2f} MB")
        logging.info("="*60)
    
    # 运行异步处理
    asyncio.run(run_processing())


if __name__ == "__main__":
    main() 