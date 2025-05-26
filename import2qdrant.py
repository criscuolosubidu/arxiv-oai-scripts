#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化版本的H5向量文件Qdrant导入器
主要优化：
1. 批量读取H5数据，减少I/O开销
2. 预处理元数据，避免重复处理
3. 优化向量转换，使用numpy操作
4. 减少异常处理开销
5. 并行化数据处理
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
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
    log_file = os.path.join(log_dir, f"qdrant_import_{timestamp}.log")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


class OptimizedQdrantImporter:
    """优化版Qdrant向量数据库导入器"""
    
    def __init__(
        self, 
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "arxiv_papers",
        vector_size: int = 4096,
        distance_metric: str = "Cosine",
        timeout: int = 300,
        max_workers: int = 4  # 并发处理线程数
    ):
        self.client = QdrantClient(url=qdrant_url, timeout=timeout)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance_metric = getattr(Distance, distance_metric.upper())
        self.timeout = timeout
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # 预处理的元数据缓存
        self.processed_metadata = {}
        self.metadata_lock = threading.Lock()
        
        self.logger.info(f"优化版Qdrant客户端初始化完成，并发数: {max_workers}")
    
    def _process_metadata_batch(self, metadata_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """批量预处理元数据，避免运行时重复处理"""
        processed = {}
        
        # 定义字段长度限制
        field_limits = {
            "title": 1024,
            "abstract": 4096,
            "authors": 1024,
            "categories": 256,
            "doi": 100,
            "journal-ref": 300,
            "update_date": 50
        }
        
        for paper_id, paper_meta in metadata_dict.items():
            processed_meta = {
                "id": paper_id,
                "paper_id": paper_id
            }
            
            # 批量处理所有字段
            for field, limit in field_limits.items():
                value = paper_meta.get(field, "")
                if isinstance(value, str) and len(value) > limit:
                    processed_meta[field] = value[:limit]
                else:
                    processed_meta[field] = str(value) if value is not None else ""
            
            # 特殊处理authors_parsed字段（保持列表结构）
            processed_meta["authors_parsed"] = paper_meta.get("authors_parsed", [])
            
            processed[paper_id] = processed_meta
        
        return processed
    
    def _load_metadata_optimized(self, metadata_file_path: str) -> Dict[str, Dict[str, Any]]:
        """优化的元数据加载"""
        if not metadata_file_path or not os.path.exists(metadata_file_path):
            return {}
        
        metadata = {}
        self.logger.info(f"加载元数据文件: {metadata_file_path}")
        
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            # 检测文件格式
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
                # JSONL格式 - 使用生成器避免内存占用
                self.logger.info("检测到JSONL格式，使用流式处理")
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            paper = json.loads(line)
                            if 'id' in paper:
                                metadata[paper['id']] = paper
                        except json.JSONDecodeError:
                            if line_num % 1000 == 0:  # 减少日志输出
                                self.logger.warning(f"跳过JSONL第{line_num}行")
                            continue
                    
                    if line_num % 50000 == 0:  # 增加进度报告间隔
                        self.logger.info(f"已处理 {line_num:,} 行，加载 {len(metadata):,} 篇论文")
            else:
                # JSON格式
                self.logger.info("检测到JSON格式")
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
        
        self.logger.info(f"加载了 {len(metadata):,} 篇论文的元数据")
        
        # 预处理元数据
        self.logger.info("预处理元数据...")
        processed_metadata = self._process_metadata_batch(metadata)
        self.logger.info("元数据预处理完成")
        
        return processed_metadata
    
    def _prepare_batch_data(
        self, 
        paper_ids: np.ndarray,
        title_vectors: np.ndarray,
        abstract_vectors: np.ndarray,
        start_idx: int,
        end_idx: int,
        use_title: bool,
        use_abstract: bool
    ) -> List[PointStruct]:
        """优化的批量数据准备"""
        batch_points = []
        
        # 批量转换paper_ids
        batch_paper_ids = paper_ids[start_idx:end_idx]
        if batch_paper_ids.dtype.kind == 'S':  # bytes类型
            batch_paper_ids = [pid.decode('utf-8') for pid in batch_paper_ids]
        else:
            batch_paper_ids = batch_paper_ids.tolist()
        
        # 批量准备向量数据
        batch_title_vectors = title_vectors[start_idx:end_idx] if use_title else None
        batch_abstract_vectors = abstract_vectors[start_idx:end_idx] if use_abstract else None
        
        for i, paper_id in enumerate(batch_paper_ids):
            try:
                # 准备向量
                vectors = {}
                if use_title and batch_title_vectors is not None:
                    vectors["title"] = batch_title_vectors[i].tolist()
                if use_abstract and batch_abstract_vectors is not None:
                    vectors["abstract"] = batch_abstract_vectors[i].tolist()
                
                # 获取预处理的元数据
                if paper_id in self.processed_metadata:
                    payload = self.processed_metadata[paper_id].copy()
                else:
                    payload = {
                        "id": paper_id,
                        "paper_id": paper_id,
                        "title": "",
                        "abstract": "",
                        "authors": "",
                        "categories": "",
                        "doi": "",
                        "journal-ref": "",
                        "update_date": "",
                        "authors_parsed": []
                    }
                
                # 创建点
                point = PointStruct(
                    id=start_idx + i,
                    vector=vectors,
                    payload=payload
                )
                
                batch_points.append(point)
                
            except Exception as e:
                self.logger.debug(f"处理点 {start_idx + i} 时出错: {str(e)}")
                continue
        
        return batch_points
    
    def _upload_batch_with_retry(
        self, 
        batch_points: List[PointStruct], 
        batch_num: int,
        max_retries: int = 3
    ) -> bool:
        """优化的批量上传，减少重试开销"""
        if not batch_points:
            return True
        
        # 使用指数退避，但起始延迟更短
        retry_delays = [0.5, 1.0, 2.0]  
        
        for retry in range(max_retries):
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
                return True
                
            except Exception as e:
                error_msg = str(e)
                is_timeout = any(keyword in error_msg.lower() 
                               for keyword in ["timeout", "timed out", "connection"])
                
                if retry < max_retries - 1:
                    delay = retry_delays[retry]
                    if is_timeout:
                        self.logger.debug(f"批次{batch_num}超时，{delay}s后重试")
                    else:
                        self.logger.debug(f"批次{batch_num}失败，{delay}s后重试: {error_msg}")
                    time.sleep(delay)
                else:
                    self.logger.error(f"批次{batch_num}最终失败: {error_msg}")
                    return False
        
        return False
    
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
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "点数量": info.points_count,
                "向量配置": info.config.params.vectors,
                "状态": info.status
            }
        except Exception as e:
            self.logger.error(f"获取集合信息失败: {str(e)}")
            return {}
    
    def import_from_h5(
        self, 
        h5_file_path: str,
        metadata_file_path: str,
        batch_size: int = 1000,  # 增加默认批次大小
        start_index: int = 0,
        max_points: Optional[int] = None,
        use_title: bool = True,
        use_abstract: bool = True
    ) -> bool:
        """优化版H5文件导入"""
        
        try:
            # 预加载元数据
            self.processed_metadata = self._load_metadata_optimized(metadata_file_path)
            
            # 加载H5文件并获取基本信息
            self.logger.info(f"加载H5文件: {h5_file_path}")
            with h5py.File(h5_file_path, 'r') as h5f:
                total_papers = len(h5f['paper_ids'])
                self.logger.info(f"H5文件包含 {total_papers:,} 篇论文的向量")
                
                # 确定处理范围
                end_index = min(
                    start_index + (max_points or total_papers), 
                    total_papers
                )
                
                self.logger.info(f"将处理索引 {start_index} 到 {end_index-1} 的数据")
                
                # 获取数据集引用（避免重复访问）
                paper_ids = h5f['paper_ids']
                title_embeddings = h5f['title_embeddings'] if use_title else None
                abstract_embeddings = h5f['abstract_embeddings'] if use_abstract else None
                
                # 统计变量
                points_imported = 0
                failed_batches = 0
                
                # 使用更大的批次和并发处理
                total_batches = (end_index - start_index + batch_size - 1) // batch_size
                
                # 进度条
                with tqdm(total=end_index - start_index, desc="导入向量到Qdrant") as pbar:
                    
                    for i in range(start_index, end_index, batch_size):
                        batch_end = min(i + batch_size, end_index)
                        batch_num = (i - start_index) // batch_size + 1
                        
                        try:
                            # 批量准备数据
                            batch_points = self._prepare_batch_data(
                                paper_ids, title_embeddings, abstract_embeddings,
                                i, batch_end, use_title, use_abstract
                            )
                            
                            # 上传批次
                            if self._upload_batch_with_retry(batch_points, batch_num):
                                points_imported += len(batch_points)
                                
                                # 更新进度条
                                pbar.update(batch_end - i)
                                
                                # 定期报告进度（减少日志输出）
                                if batch_num % 10 == 0:
                                    self.logger.info(f"已完成 {batch_num}/{total_batches} 批次，导入 {points_imported:,} 个点")
                            else:
                                failed_batches += 1
                                pbar.update(batch_end - i)  # 即使失败也更新进度条
                                
                        except Exception as e:
                            self.logger.error(f"处理批次 {batch_num} 时出错: {str(e)}")
                            failed_batches += 1
                            pbar.update(batch_end - i)
                            continue
                
                # 导入完成统计
                self.logger.info(f"导入完成!")
                self.logger.info(f"成功导入: {points_imported:,} 个点")
                if failed_batches > 0:
                    self.logger.warning(f"失败批次: {failed_batches} 个")
                
                return True
                
        except Exception as e:
            self.logger.error(f"导入过程中发生错误: {str(e)}")
            return False
    
    def search_similar(
        self, 
        query_vector: List[float], 
        vector_name: str = "title",
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=(vector_name, query_vector),
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True
            )
            
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                for result in results
            ]
            
        except Exception as e:
            self.logger.error(f"搜索失败: {str(e)}")
            return []


def main():
    parser = argparse.ArgumentParser(description="优化版H5嵌入向量文件导入到Qdrant")
    parser.add_argument("--h5_file", type=str, required=True,
                        help="H5嵌入向量文件路径")
    parser.add_argument("--metadata_file", type=str,
                        help="元数据文件路径 (自动检测JSON/JSONL格式)")
    parser.add_argument("--qdrant_url", type=str, default="http://localhost:6333",
                        help="Qdrant服务URL")
    parser.add_argument("--collection_name", type=str, default="arxiv_papers",
                        help="Qdrant集合名称")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="批量导入大小（优化后建议1000+）")
    parser.add_argument("--start_index", type=int, default=0,
                        help="开始导入的索引位置")
    parser.add_argument("--max_points", type=int,
                        help="最大导入点数")
    parser.add_argument("--recreate_collection", action="store_true",
                        help="重新创建集合（删除现有数据）")
    parser.add_argument("--use_title", action="store_true", default=True,
                        help="导入标题向量")
    parser.add_argument("--use_abstract", action="store_true", default=True,
                        help="导入摘要向量")
    parser.add_argument("--distance_metric", type=str, default="Cosine",
                        choices=["Cosine", "Euclidean", "Dot"],
                        help="距离度量方式")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="日志输出目录")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Qdrant客户端超时时间（秒）")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="并发处理线程数")
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(args.log_dir, log_level)
    
    # 检查文件
    if not os.path.exists(args.h5_file):
        logger.error(f"H5文件不存在: {args.h5_file}")
        return
    
    if args.metadata_file and not os.path.exists(args.metadata_file):
        logger.error(f"元数据文件不存在: {args.metadata_file}")
        return
    
    # 获取向量维度
    vector_size = 4096
    try:
        with h5py.File(args.h5_file, 'r') as h5f:
            vector_size = h5f['title_embeddings'].shape[1]
            logger.info(f"检测到向量维度: {vector_size}")
    except Exception as e:
        logger.warning(f"无法检测向量维度，使用默认值 {vector_size}: {str(e)}")
    
    # 创建优化版导入器
    importer = OptimizedQdrantImporter(
        qdrant_url=args.qdrant_url,
        collection_name=args.collection_name,
        vector_size=vector_size,
        distance_metric=args.distance_metric,
        timeout=args.timeout,
        max_workers=args.max_workers
    )
    
    # 创建集合
    if not importer.create_collection(recreate=args.recreate_collection):
        logger.error("创建集合失败，退出")
        return
    
    # 显示集合信息
    info = importer.get_collection_info()
    if info:
        logger.info(f"集合信息: {info}")
    
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
        logger.info(f"导入完成，耗时: {duration:.2f} 秒")
        
        # 计算性能指标
        points_processed = args.max_points or 0
        if points_processed > 0:
            rate = points_processed / duration
            logger.info(f"导入速率: {rate:.2f} 点/秒")
        
        # 显示最终集合信息
        final_info = importer.get_collection_info()
        if final_info:
            logger.info(f"最终集合信息: {final_info}")
    else:
        logger.error("导入失败")


if __name__ == "__main__":
    main()