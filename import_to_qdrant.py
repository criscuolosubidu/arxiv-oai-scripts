#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多进程并行版本：将H5格式的嵌入向量文件导入到Qdrant向量数据库
使用多进程并行上传，显著提升导入性能

性能优化特点：
1. 多进程并行处理，充分利用多核CPU
2. 每个进程独立处理数据片段
3. 进程间负载均衡
4. 实时进度监控和错误收集
"""

import os
import json
import argparse
import logging
import h5py
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple
import traceback

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, 
    )
except ImportError:
    print("请安装 qdrant-client: pip install qdrant-client")
    exit(1)


def setup_logger(log_dir: str = "logs", log_level: int = logging.INFO) -> logging.Logger:
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"qdrant_import_mp_{timestamp}.log")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - [PID:%(process)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_metadata(metadata_file_path: str) -> Dict[str, Dict[str, Any]]:
    """加载元数据文件"""
    metadata = {}
    if not metadata_file_path or not os.path.exists(metadata_file_path):
        return metadata
    
    with open(metadata_file_path, 'r', encoding='utf-8') as f:
        # 判断文件格式
        first_line = f.readline().strip()
        f.seek(0)
        
        is_jsonl = False
        if first_line:
            try:
                first_obj = json.loads(first_line)
                if isinstance(first_obj, dict) and 'id' in first_obj:
                    second_line = f.readline().strip()
                    if second_line:
                        try:
                            json.loads(second_line)
                            is_jsonl = True
                        except json.JSONDecodeError:
                            pass
                    f.seek(0)
            except json.JSONDecodeError:
                pass
        
        if is_jsonl:
            # JSONL格式
            for line in f:
                line = line.strip()
                if line:
                    try:
                        paper = json.loads(line)
                        if 'id' in paper:
                            metadata[paper['id']] = paper
                    except json.JSONDecodeError:
                        continue
        else:
            # JSON格式
            try:
                metadata_json = json.load(f)
                
                if isinstance(metadata_json, list):
                    papers = metadata_json
                elif isinstance(metadata_json, dict):
                    if 'papers' in metadata_json:
                        papers = metadata_json.get('papers', [])
                    elif 'id' in metadata_json:
                        papers = [metadata_json]
                    else:
                        papers = metadata_json.get('papers', [])
                else:
                    papers = []
                
                for paper in papers:
                    if isinstance(paper, dict) and 'id' in paper:
                        metadata[paper['id']] = paper
                        
            except json.JSONDecodeError:
                pass
    
    return metadata


def worker_process(
    process_id: int,
    h5_file_path: str,
    metadata: Dict[str, Dict[str, Any]],
    start_idx: int,
    end_idx: int,
    qdrant_url: str,
    collection_name: str,
    batch_size: int,
    use_title: bool,
    use_abstract: bool,
    timeout: int,
    progress_queue: mp.Queue,
    result_queue: mp.Queue
) -> None:
    """工作进程函数"""
    try:
        # 创建进程专用的Qdrant客户端
        client = QdrantClient(url=qdrant_url, timeout=timeout)
        
        # 打开H5文件
        with h5py.File(h5_file_path, 'r') as h5f:
            paper_ids = h5f['paper_ids']
            title_embeddings = h5f['title_embeddings']
            abstract_embeddings = h5f['abstract_embeddings']
            
            points_imported = 0
            failed_imports = 0
            
            # 批量处理
            for i in range(start_idx, end_idx, batch_size):
                batch_end = min(i + batch_size, end_idx)
                batch_points = []
                
                for idx in range(i, batch_end):
                    try:
                        # 获取论文ID
                        paper_id = paper_ids[idx]
                        if isinstance(paper_id, bytes):
                            paper_id = paper_id.decode('utf-8')
                        
                        # 准备向量
                        vectors = {}
                        if use_title:
                            vectors["title"] = title_embeddings[idx].tolist()
                        if use_abstract:
                            vectors["abstract"] = abstract_embeddings[idx].tolist()
                        
                        # 准备载荷
                        payload = {
                            "id": paper_id,
                            "paper_id": paper_id
                        }
                        
                        # 添加元数据
                        if paper_id in metadata:
                            paper_meta = metadata[paper_id]
                            payload.update({
                                "title": paper_meta.get("title", ""),
                                "abstract": paper_meta.get("abstract", ""),
                                "authors": paper_meta.get("authors", ""),
                                "categories": paper_meta.get("categories", ""),
                                "doi": paper_meta.get("doi", ""),
                                "journal-ref": paper_meta.get("journal-ref", ""),
                                "update_date": paper_meta.get("update_date", ""),
                                "authors_parsed": paper_meta.get("authors_parsed", [])
                            })
                        
                        # 创建点
                        point = PointStruct(
                            id=idx,
                            vector=vectors,
                            payload=payload
                        )
                        
                        batch_points.append(point)
                        
                    except Exception as e:
                        failed_imports += 1
                        continue
                
                # 上传批次（带重试）
                if batch_points:
                    max_retries = 3
                    retry_delay = 1
                    
                    for retry in range(max_retries):
                        try:
                            client.upsert(
                                collection_name=collection_name,
                                points=batch_points
                            )
                            points_imported += len(batch_points)
                            break
                        except Exception as e:
                            if retry < max_retries - 1:
                                time.sleep(retry_delay)
                                retry_delay *= 2
                            else:
                                failed_imports += len(batch_points)
                
                # 报告进度
                progress_queue.put({
                    'process_id': process_id,
                    'imported': len(batch_points),
                    'failed': 0 if batch_points else len(range(i, batch_end))
                })
        
        # 报告最终结果
        result_queue.put({
            'process_id': process_id,
            'success': True,
            'points_imported': points_imported,
            'failed_imports': failed_imports,
            'error': None
        })
        
    except Exception as e:
        # 报告错误
        result_queue.put({
            'process_id': process_id,
            'success': False,
            'points_imported': 0,
            'failed_imports': 0,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


class MultiProcessQdrantImporter:
    """多进程Qdrant向量数据库导入器"""
    
    def __init__(
        self, 
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "arxiv_papers",
        vector_size: int = 4096,
        distance_metric: str = "Cosine",
        timeout: int = 300,
        num_processes: int = None
    ):
        self.qdrant_url = qdrant_url  # 保存URL以便传递给子进程
        self.client = QdrantClient(url=qdrant_url, timeout=timeout)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance_metric = getattr(Distance, distance_metric.upper())
        self.timeout = timeout
        self.num_processes = num_processes or mp.cpu_count()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"多进程导入器初始化完成，进程数: {self.num_processes}")
    
    def create_collection(self, recreate: bool = False) -> bool:
        """创建Qdrant集合"""
        try:
            collections = self.client.get_collections()
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            
            if collection_exists:
                if recreate:
                    self.logger.info(f"删除现有集合: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    self.logger.info(f"集合 {self.collection_name} 已存在，将追加数据")
                    return True
            
            self.logger.info(f"创建新集合: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "title": VectorParams(
                        size=self.vector_size,
                        distance=self.distance_metric
                    ),
                    "abstract": VectorParams(
                        size=self.vector_size,
                        distance=self.distance_metric
                    )
                }
            )
            
            self.logger.info(f"集合 {self.collection_name} 创建成功")
            return True
            
        except Exception as e:
            self.logger.error(f"创建集合失败: {str(e)}")
            return False
    
    def import_from_h5(
        self, 
        h5_file_path: str,
        metadata_file_path: str,
        batch_size: int = 500,
        start_index: int = 0,
        max_points: Optional[int] = None,
        use_title: bool = True,
        use_abstract: bool = True
    ) -> bool:
        """多进程导入H5文件到Qdrant"""
        
        try:
            # 获取数据总量
            with h5py.File(h5_file_path, 'r') as h5f:
                total_papers = len(h5f['paper_ids'])
                self.logger.info(f"H5文件包含 {total_papers:,} 篇论文的向量")
            
            # 确定处理范围
            end_index = min(
                start_index + (max_points or total_papers), 
                total_papers
            )
            total_to_process = end_index - start_index
            
            self.logger.info(f"将处理索引 {start_index} 到 {end_index-1} 的数据")
            
            # 加载元数据
            self.logger.info("加载元数据...")
            metadata = load_metadata(metadata_file_path)
            self.logger.info(f"加载了 {len(metadata)} 篇论文的元数据")
            
            # 计算每个进程的数据范围
            chunk_size = total_to_process // self.num_processes
            if chunk_size == 0:
                chunk_size = 1
                self.num_processes = total_to_process
            
            process_ranges = []
            for i in range(self.num_processes):
                proc_start = start_index + i * chunk_size
                if i == self.num_processes - 1:
                    proc_end = end_index  # 最后一个进程处理剩余所有数据
                else:
                    proc_end = proc_start + chunk_size
                
                if proc_start < end_index:
                    process_ranges.append((proc_start, proc_end))
            
            self.logger.info(f"数据分片: {len(process_ranges)} 个进程")
            for i, (start, end) in enumerate(process_ranges):
                self.logger.info(f"进程 {i}: 索引 {start} 到 {end-1} ({end-start} 个点)")
            
            # 创建进程间通信队列
            progress_queue = mp.Queue()
            result_queue = mp.Queue()
            
            # 启动工作进程
            processes = []
            for i, (proc_start, proc_end) in enumerate(process_ranges):
                p = mp.Process(
                    target=worker_process,
                    args=(
                        i, h5_file_path, metadata, proc_start, proc_end,
                        self.qdrant_url, self.collection_name,
                        batch_size, use_title, use_abstract, self.timeout,
                        progress_queue, result_queue
                    )
                )
                p.start()
                processes.append(p)
            
            # 监控进度
            total_imported = 0
            total_failed = 0
            process_stats = {i: {'imported': 0, 'failed': 0} for i in range(len(processes))}
            
            # 使用tqdm显示总体进度
            with tqdm(total=total_to_process, desc="多进程导入进度") as pbar:
                # 收集进度更新
                active_processes = len(processes)
                while active_processes > 0:
                    try:
                        # 检查进度更新
                        try:
                            progress_update = progress_queue.get(timeout=1)
                            process_id = progress_update['process_id']
                            imported = progress_update['imported']
                            failed = progress_update['failed']
                            
                            process_stats[process_id]['imported'] += imported
                            process_stats[process_id]['failed'] += failed
                            
                            pbar.update(imported + failed)
                            
                        except:
                            pass  # 超时，继续检查进程状态
                        
                        # 检查是否有进程完成
                        for p in processes[:]:
                            if not p.is_alive():
                                processes.remove(p)
                                active_processes -= 1
                    
                    except KeyboardInterrupt:
                        self.logger.info("收到中断信号，正在停止所有进程...")
                        for p in processes:
                            p.terminate()
                        break
            
            # 等待所有进程完成
            for p in processes:
                p.join()
            
            # 收集最终结果
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())
            
            # 统计结果
            for result in results:
                if result['success']:
                    total_imported += result['points_imported']
                    total_failed += result['failed_imports']
                    self.logger.info(f"进程 {result['process_id']} 完成: "
                                   f"导入 {result['points_imported']}, "
                                   f"失败 {result['failed_imports']}")
                else:
                    self.logger.error(f"进程 {result['process_id']} 失败: {result['error']}")
                    if 'traceback' in result:
                        self.logger.debug(f"错误详情:\n{result['traceback']}")
            
            self.logger.info(f"多进程导入完成!")
            self.logger.info(f"总计导入: {total_imported:,} 个点")
            if total_failed > 0:
                self.logger.warning(f"总计失败: {total_failed:,} 个点")
            
            return True
            
        except Exception as e:
            self.logger.error(f"多进程导入过程中发生错误: {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(description="多进程并行将H5嵌入向量文件导入到Qdrant")
    parser.add_argument("--h5_file", type=str, required=True,
                        help="H5嵌入向量文件路径")
    parser.add_argument("--metadata_file", type=str,
                        help="元数据文件路径")
    parser.add_argument("--qdrant_url", type=str, default="http://localhost:6333",
                        help="Qdrant服务URL")
    parser.add_argument("--collection_name", type=str, default="arxiv_papers",
                        help="Qdrant集合名称")
    parser.add_argument("--batch_size", type=int, default=500,
                        help="批量导入大小")
    parser.add_argument("--start_index", type=int, default=0,
                        help="开始导入的索引位置")
    parser.add_argument("--max_points", type=int,
                        help="最大导入点数")
    parser.add_argument("--recreate_collection", action="store_true",
                        help="重新创建集合")
    parser.add_argument("--use_title", action="store_true", default=True,
                        help="导入标题向量")
    parser.add_argument("--use_abstract", action="store_true", default=True,
                        help="导入摘要向量")
    parser.add_argument("--distance_metric", type=str, default="Cosine",
                        choices=["Cosine", "Euclidean", "Dot"],
                        help="距离度量方式")
    parser.add_argument("--num_processes", type=int,
                        help="并行进程数（默认为CPU核心数）")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Qdrant客户端超时时间（秒）")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="日志输出目录")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(args.log_dir, log_level)
    
    # 检查文件
    if not os.path.exists(args.h5_file):
        logger.error(f"H5文件不存在: {args.h5_file}")
        return
    
    # 获取向量维度
    vector_size = 4096
    try:
        with h5py.File(args.h5_file, 'r') as h5f:
            vector_size = h5f['title_embeddings'].shape[1]
            logger.info(f"检测到向量维度: {vector_size}")
    except Exception as e:
        logger.warning(f"无法检测向量维度，使用默认值 {vector_size}")
    
    # 创建多进程导入器
    importer = MultiProcessQdrantImporter(
        qdrant_url=args.qdrant_url,
        collection_name=args.collection_name,
        vector_size=vector_size,
        distance_metric=args.distance_metric,
        timeout=args.timeout,
        num_processes=args.num_processes
    )
    
    # 创建集合
    if not importer.create_collection(recreate=args.recreate_collection):
        logger.error("创建集合失败")
        return
    
    # 开始导入
    start_time = time.time()
    success = importer.import_from_h5(
        h5_file_path=args.h5_file,
        metadata_file_path=args.metadata_file,
        batch_size=args.batch_size,
        start_index=args.start_index,
        max_points=args.max_points,
        use_title=args.use_title,
        use_abstract=args.use_abstract
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    if success:
        logger.info(f"多进程导入完成，耗时: {duration:.2f} 秒")
    else:
        logger.error("多进程导入失败")


if __name__ == "__main__":
    main()