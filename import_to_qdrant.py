#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将H5格式的嵌入向量文件导入到Qdrant向量数据库
支持批量导入、断点续传、进度监控等功能

支持的元数据格式（自动检测）：
1. JSON格式: 
   - 论文列表: [{"id": "...", "title": "..."}, ...]
   - 包装对象: {"papers": [{"id": "...", "title": "..."}, ...]}
   - 单个论文: {"id": "...", "title": "..."}

2. JSONL格式: 每行一个JSON对象
   {"id": "0704.0001", "title": "...", "abstract": "..."}
   {"id": "0704.0002", "title": "...", "abstract": "..."}

支持的元数据字段：
- id: 论文ID (如 "0704.0001")
- submitter: 提交者
- authors: 作者列表字符串
- title: 论文标题
- comments: 评论信息
- journal-ref: 期刊引用
- doi: DOI标识符
- report-no: 报告编号
- categories: 分类标签
- license: 许可证信息
- abstract: 论文摘要
- versions: 版本历史列表
- update_date: 更新日期
- authors_parsed: 解析后的作者信息列表
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


class QdrantImporter:
    """Qdrant向量数据库导入器"""
    
    def __init__(
        self, 
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "arxiv_papers",
        vector_size: int = 4096,
        distance_metric: str = "Cosine",
        timeout: int = 300  # 5分钟超时
    ):
        # 创建Qdrant客户端，设置更长的超时时间
        self.client = QdrantClient(
            url=qdrant_url,
            timeout=timeout  # 设置超时时间
        )
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance_metric = getattr(Distance, distance_metric.upper())
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Qdrant客户端初始化完成，超时时间: {timeout}秒")
    
    def _process_metadata_field(self, value: Any, max_length: Optional[int] = None) -> Any:
        """处理元数据字段，确保数据类型适合Qdrant存储"""
        if value is None:
            return ""
        
        if isinstance(value, (list, dict)):
            # 对于复杂数据结构，转换为JSON字符串或保持原样，这里因为qdrant支持，所以无需转换
            return value
        
        if isinstance(value, str) and max_length:
            return value[:max_length]
        
        return str(value) if not isinstance(value, (int, float, bool)) else value
        
    def create_collection(self, recreate: bool = False) -> bool:
        """创建Qdrant集合"""
        try:
            # 检查集合是否存在
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
            
            # 创建新集合
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
        batch_size: int = 500,
        start_index: int = 0,
        max_points: Optional[int] = None,
        use_title: bool = True,
        use_abstract: bool = True
    ) -> bool:
        """从H5文件导入向量到Qdrant"""
        
        try:
            # 加载H5文件
            self.logger.info(f"加载H5文件: {h5_file_path}")
            with h5py.File(h5_file_path, 'r') as h5f:
                paper_ids = h5f['paper_ids'][:]
                title_embeddings = h5f['title_embeddings']
                abstract_embeddings = h5f['abstract_embeddings']
                
                total_papers = len(paper_ids)
                self.logger.info(f"H5文件包含 {total_papers:,} 篇论文的向量")
                
                # 确定处理范围
                end_index = min(
                    start_index + (max_points or total_papers), 
                    total_papers
                )
                
                self.logger.info(f"将处理索引 {start_index} 到 {end_index-1} 的数据")
                
                # 加载元数据
                metadata = {}
                if metadata_file_path and os.path.exists(metadata_file_path):
                    self.logger.info(f"加载元数据文件: {metadata_file_path}")
                    
                    with open(metadata_file_path, 'r', encoding='utf-8') as f:
                        # 先读取第一行来判断格式
                        first_line = f.readline().strip()
                        f.seek(0)  # 重置文件指针
                        
                        # 尝试判断文件格式
                        is_jsonl = False
                        if first_line:
                            try:
                                # 尝试解析第一行为JSON对象
                                first_obj = json.loads(first_line)
                                if isinstance(first_obj, dict) and 'id' in first_obj:
                                    # 检查是否还有第二行
                                    second_line = f.readline().strip()
                                    if second_line:
                                        try:
                                            # 如果第二行也能解析为JSON对象，则判断为JSONL
                                            json.loads(second_line)
                                            is_jsonl = True
                                        except json.JSONDecodeError:
                                            # 第二行解析失败，可能是JSON格式
                                            pass
                                    f.seek(0)  # 重置文件指针
                            except json.JSONDecodeError:
                                # 第一行不是有效JSON，尝试作为完整JSON文件解析
                                pass
                        
                        if is_jsonl:
                            # JSONL格式：每行一个JSON对象
                            self.logger.info("检测到JSONL格式")
                            line_count = 0
                            error_count = 0
                            
                            # 先计算总行数用于进度显示
                            f.seek(0)
                            total_lines = sum(1 for line in f if line.strip())
                            f.seek(0)
                            
                            self.logger.info(f"JSONL文件包含 {total_lines:,} 行")
                            
                            for line_num, line in enumerate(f, 1):
                                line = line.strip()
                                if line:
                                    try:
                                        paper = json.loads(line)
                                        if 'id' in paper:
                                            metadata[paper['id']] = paper
                                            line_count += 1
                                        else:
                                            self.logger.warning(f"第{line_num}行缺少'id'字段")
                                            error_count += 1
                                    except json.JSONDecodeError as e:
                                        self.logger.warning(f"解析JSONL第{line_num}行失败: {str(e)}")
                                        error_count += 1
                                        continue
                                
                                # 每10000行显示一次进度
                                if line_num % 10000 == 0:
                                    self.logger.info(f"已处理 {line_num:,}/{total_lines:,} 行，成功加载 {line_count:,} 篇论文")
                            
                            self.logger.info(f"从JSONL文件加载了 {len(metadata)} 篇论文的元数据")
                            if error_count > 0:
                                self.logger.warning(f"跳过了 {error_count} 行错误数据")
                        else:
                            # JSON格式
                            self.logger.info("检测到JSON格式")
                            try:
                                metadata_json = json.load(f)
                                
                                # 支持多种JSON格式
                                if isinstance(metadata_json, list):
                                    # 直接是论文列表
                                    papers = metadata_json
                                elif isinstance(metadata_json, dict):
                                    if 'papers' in metadata_json:
                                        # 包含papers字段的对象
                                        papers = metadata_json.get('papers', [])
                                    elif 'id' in metadata_json:
                                        # 单个论文对象
                                        papers = [metadata_json]
                                    else:
                                        # 其他格式，尝试获取papers字段
                                        papers = metadata_json.get('papers', [])
                                else:
                                    self.logger.error("不支持的JSON格式")
                                    papers = []
                                
                                # 构建ID到元数据的映射
                                for paper in papers:
                                    if isinstance(paper, dict) and 'id' in paper:
                                        metadata[paper['id']] = paper
                                
                                self.logger.info(f"从JSON文件加载了 {len(metadata)} 篇论文的元数据")
                                
                            except json.JSONDecodeError as e:
                                self.logger.error(f"解析JSON文件失败: {str(e)}")
                                return False
                
                # 批量处理和导入
                points_imported = 0
                failed_imports = 0
                
                for i in tqdm(
                    range(start_index, end_index, batch_size),
                    desc="导入向量到Qdrant"
                ):
                    batch_end = min(i + batch_size, end_index)
                    
                    # 准备批次数据
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
                            
                            # 准备载荷（元数据）
                            payload = {
                                "id": paper_id,  # 确保论文ID包含在载荷中
                                "paper_id": paper_id  # 保持向后兼容
                            }
                            
                            # 添加详细元数据（如果可用）
                            if paper_id in metadata:
                                paper_meta = metadata[paper_id]
                                
                                # 记录第一个处理的论文的元数据字段（用于调试）
                                if idx == start_index:
                                    available_fields = list(paper_meta.keys())
                                    self.logger.info(f"检测到的元数据字段: {available_fields}")
                                
                                payload.update({
                                    # 基本信息
                                    "title": self._process_metadata_field(paper_meta.get("title", ""), 1024),
                                    "abstract": self._process_metadata_field(paper_meta.get("abstract", ""), 4096),
                                    "authors": self._process_metadata_field(paper_meta.get("authors", ""), 1024),
                                    
                                    # 分类和标识
                                    "categories": self._process_metadata_field(paper_meta.get("categories", ""), 256),
                                    "doi": self._process_metadata_field(paper_meta.get("doi", ""), 100),
                                    
                                    # 期刊和出版信息
                                    "journal-ref": self._process_metadata_field(paper_meta.get("journal-ref", ""), 300),
                                    
                                    # 时间信息
                                    "update_date": self._process_metadata_field(paper_meta.get("update_date", ""), 50),
                                    
                                    # 解析后的作者信息（保持为列表结构）
                                    "authors_parsed": self._process_metadata_field(paper_meta.get("authors_parsed", []))
                                })
                            
                            # 创建点
                            point = PointStruct(
                                id=idx,  # 使用索引作为ID
                                vector=vectors,
                                payload=payload
                            )
                            
                            batch_points.append(point)
                            
                        except Exception as e:
                            self.logger.warning(f"处理索引 {idx} 的数据时出错: {str(e)}")
                            failed_imports += 1
                            continue
                    
                    # 批量上传到Qdrant（带重试机制）
                    if batch_points:
                        upload_success = False
                        max_retries = 3
                        retry_delay = 1  # 秒
                        
                        for retry in range(max_retries):
                            try:
                                self.logger.debug(f"尝试上传批次 {i//batch_size + 1}，重试次数: {retry + 1}/{max_retries}")
                                
                                self.client.upsert(
                                    collection_name=self.collection_name,
                                    points=batch_points
                                )
                                
                                points_imported += len(batch_points)
                                upload_success = True
                                
                                # 每1000个点记录一次进度
                                if points_imported % 1000 == 0:
                                    self.logger.info(f"已导入 {points_imported:,} 个点")
                                
                                break  # 成功上传，跳出重试循环
                                
                            except Exception as e:
                                error_msg = str(e)
                                if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
                                    self.logger.warning(f"上传超时 (重试 {retry + 1}/{max_retries}): {error_msg}")
                                else:
                                    self.logger.warning(f"上传失败 (重试 {retry + 1}/{max_retries}): {error_msg}")
                                
                                if retry < max_retries - 1:  # 不是最后一次重试
                                    self.logger.info(f"等待 {retry_delay} 秒后重试...")
                                    time.sleep(retry_delay)
                                    retry_delay *= 2  # 指数退避
                                else:
                                    self.logger.error(f"批量上传最终失败，跳过 {len(batch_points)} 个点")
                                    self.logger.error(f"建议：如果持续超时，请尝试减小 --batch_size 参数（当前: {batch_size}）")
                                    failed_imports += len(batch_points)
                        
                        if not upload_success:
                            continue
                
                # 导入完成统计
                self.logger.info(f"导入完成!")
                self.logger.info(f"成功导入: {points_imported:,} 个点")
                if failed_imports > 0:
                    self.logger.warning(f"导入失败: {failed_imports:,} 个点")
                
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
    parser = argparse.ArgumentParser(description="将H5嵌入向量文件导入到Qdrant")
    parser.add_argument("--h5_file", type=str, required=True,
                        help="H5嵌入向量文件路径")
    parser.add_argument("--metadata_file", type=str,
                        help="元数据文件路径 (自动检测JSON/JSONL格式)")
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
                        help="Qdrant客户端超时时间（秒），默认300秒")
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(args.log_dir, log_level)
    
    # 检查文件是否存在
    if not os.path.exists(args.h5_file):
        logger.error(f"H5文件不存在: {args.h5_file}")
        return
    
    if args.metadata_file and not os.path.exists(args.metadata_file):
        logger.error(f"元数据文件不存在: {args.metadata_file}")
        return
    
    # 获取向量维度
    vector_size = 4096  # 默认值
    try:
        with h5py.File(args.h5_file, 'r') as h5f:
            vector_size = h5f['title_embeddings'].shape[1]
            logger.info(f"检测到向量维度: {vector_size}")
    except Exception as e:
        logger.warning(f"无法检测向量维度，使用默认值 {vector_size}: {str(e)}")
    
    # 创建导入器
    importer = QdrantImporter(
        qdrant_url=args.qdrant_url,
        collection_name=args.collection_name,
        vector_size=vector_size,
        distance_metric=args.distance_metric,
        timeout=args.timeout
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
        
        # 显示最终集合信息
        final_info = importer.get_collection_info()
        if final_info:
            logger.info(f"最终集合信息: {final_info}")
    else:
        logger.error("导入失败")


if __name__ == "__main__":
    main()

# 支持的元数据格式示例（自动检测）：

# 1. JSON格式 - 论文列表:
# [
#   {
#     "id": "0704.0001",
#     "submitter": "Pavel Nadolsky",
#     "authors": "C. Bal\\'azs, E. L. Berger, P. M. Nadolsky, C.-P. Yuan",
#     "title": "Calculation of prompt diphoton production cross sections at Tevatron and LHC energies",
#     "abstract": "A fully differential calculation..."
#   },
#   {"id": "0704.0002", "title": "Another paper", "abstract": "..."}
# ]

# 2. JSON格式 - 包装对象:
# {
#   "papers": [
#     {"id": "0704.0001", "title": "Paper 1", "abstract": "..."},
#     {"id": "0704.0002", "title": "Paper 2", "abstract": "..."}
#   ]
# }

# 3. JSON格式 - 单个论文:
# {
#   "id": "0704.0001",
#   "submitter": "Pavel Nadolsky",
#   "title": "Calculation of prompt diphoton production cross sections",
#   "abstract": "A fully differential calculation..."
# }

# 4. JSONL格式 - 每行一个JSON对象:
# {"id": "0704.0001", "submitter": "Pavel Nadolsky", "title": "Paper 1", "abstract": "..."}
# {"id": "0704.0002", "submitter": "Another Author", "title": "Paper 2", "abstract": "..."}
# {"id": "0704.0003", "submitter": "Third Author", "title": "Paper 3", "abstract": "..."}