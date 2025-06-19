#!/usr/bin/env python3
"""
基于H5文件的GPU加速向量搜索工具
使用Faiss库进行高效的向量相似度计算，支持GPU加速
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
from typing import List, Tuple, Dict, Optional
import gc

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("警告: Faiss库未安装，将使用CPU版本的相似度计算")
    print("安装命令: pip install faiss-cpu 或 pip install faiss-gpu")

# 嵌入使用的instruct（与生成脚本保持一致）
config = {
    "prompts": {
        "query_with_passage": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        "query_with_title": "Instruct: Given a web search query, retrieve relevant titles that answer the query\nQuery:",
        "query_with_abstract": "Instruct: Given a web search query, retrieve relevant abstracts that answer the query\nQuery:",
        "query_with_arxiv_paper_abstract": "Instruct: Given a web search query, retrieve relevant arxiv paper abstracts that answer the query\nQuery:",
        "query_with_arxiv_paper_title": "Instruct: Given a web search query, retrieve relevant arxiv paper titles that answer the query\nQuery:",
        "title": "Title: ",
        "document": "Passage: ",
        "abstract": "Abstract: ",
        "arxiv_paper_abstract": "Arxiv Paper Abstract: ",
        "arxiv_paper_title": "Arxiv Paper Title: "
    },
    "default_prompt_name": None,
    "similarity_fn_name": "cosine"
}


def setup_logger(log_level=logging.INFO):
    """设置日志记录器"""
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    if logger.handlers:
        logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger


def call_vllm_service_sync(text: str, api_url: str, prompt_name: Optional[str] = None, 
                          model_name: str = "qwen3-embeddings-8b", max_retries: int = 3, 
                          retry_delay: int = 1) -> np.ndarray:
    """同步调用 vLLM OpenAI 兼容 Embedding API 获取文本嵌入向量"""
    if prompt_name:
        text = prompt_name + text
    
    payload = {
        "input": text,
        "model": model_name,
        "encoding_format": "float"
    }
    
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                resp_json = response.json()
                embedding = np.array(resp_json["data"][0]["embedding"], dtype=np.float32)  # 使用float32以兼容faiss
                return embedding
            elif response.status_code == 429:
                time.sleep(retry_delay * (2 ** attempt))
                continue
            else:
                logging.error(f"vLLM服务错误，状态码: {response.status_code}")
                raise ValueError(f"vLLM服务返回错误，状态码: {response.status_code}")
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
            else:
                raise ValueError(f"调用 vLLM 服务失败: {str(e)}")
    
    raise ValueError(f"达到最大重试次数 ({max_retries})")


def cosine_similarity_batch_numpy(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """使用numpy进行批量余弦相似度计算（备用方案）"""
    # 归一化查询向量
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    
    # 归一化所有嵌入向量
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # 计算余弦相似度
    similarities = np.dot(embeddings_norm, query_norm)
    
    return similarities


class GPUEmbeddingSearcher:
    """GPU加速的嵌入向量搜索器"""
    
    def __init__(self, h5_file_path: str, metadata_file_path: Optional[str] = None, 
                 use_gpu: bool = True, gpu_id: int = 0):
        self.h5_file_path = h5_file_path
        self.metadata_file_path = metadata_file_path
        self.use_gpu = use_gpu and FAISS_AVAILABLE
        self.gpu_id = gpu_id
        
        self.h5_file = None
        self.metadata = None
        self.paper_ids = None
        self.title_embeddings = None
        self.abstract_embeddings = None
        self.embedding_dim = None
        
        # Faiss索引
        self.title_index = None
        self.abstract_index = None
        
        self._load_data()
        self._build_faiss_indices()
    
    def _load_data(self):
        """加载H5文件和元数据"""
        logging.info(f"正在加载H5文件: {self.h5_file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(self.h5_file_path):
            raise FileNotFoundError(f"H5文件不存在: {self.h5_file_path}")
        
        # 打开H5文件
        self.h5_file = h5py.File(self.h5_file_path, 'r')
        
        # 加载数据集
        self.paper_ids = self.h5_file['paper_ids'][:]
        
        # 转换为float32以兼容faiss
        self.title_embeddings = self.h5_file['title_embeddings'][:].astype(np.float32)
        self.abstract_embeddings = self.h5_file['abstract_embeddings'][:].astype(np.float32)
        
        self.embedding_dim = self.title_embeddings.shape[1]
        
        logging.info(f"成功加载 {len(self.paper_ids)} 篇论文的嵌入向量")
        logging.info(f"嵌入维度: {self.embedding_dim}")
        logging.info(f"标题嵌入形状: {self.title_embeddings.shape}")
        logging.info(f"摘要嵌入形状: {self.abstract_embeddings.shape}")
        
        # 加载元数据（如果提供）
        if self.metadata_file_path and os.path.exists(self.metadata_file_path):
            with open(self.metadata_file_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                logging.info(f"成功加载元数据文件: {self.metadata_file_path}")
    
    def _build_faiss_indices(self):
        """构建Faiss索引"""
        if not FAISS_AVAILABLE:
            logging.warning("Faiss不可用，将使用numpy进行相似度计算")
            return
        
        logging.info("正在构建Faiss索引...")
        
        # 检查GPU可用性
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                gpu_available = True
                logging.info(f"使用GPU {self.gpu_id} 进行向量搜索")
            except Exception as e:
                logging.warning(f"GPU初始化失败，将使用CPU: {str(e)}")
                gpu_available = False
                self.use_gpu = False
        else:
            gpu_available = False
            logging.info("使用CPU进行向量搜索")
        
        # 创建索引
        # 使用内积索引（对于归一化向量，内积等于余弦相似度）
        if gpu_available:
            # GPU索引
            res = faiss.StandardGpuResources()
            
            # 标题索引
            cpu_title_index = faiss.IndexFlatIP(self.embedding_dim)
            self.title_index = faiss.index_cpu_to_gpu(res, self.gpu_id, cpu_title_index)
            
            # 摘要索引
            cpu_abstract_index = faiss.IndexFlatIP(self.embedding_dim)
            self.abstract_index = faiss.index_cpu_to_gpu(res, self.gpu_id, cpu_abstract_index)
        else:
            # CPU索引
            self.title_index = faiss.IndexFlatIP(self.embedding_dim)
            self.abstract_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # 归一化向量（对于余弦相似度）
        title_embeddings_norm = self.title_embeddings.copy()
        abstract_embeddings_norm = self.abstract_embeddings.copy()
        
        faiss.normalize_L2(title_embeddings_norm)
        faiss.normalize_L2(abstract_embeddings_norm)
        
        # 添加向量到索引
        self.title_index.add(title_embeddings_norm)
        self.abstract_index.add(abstract_embeddings_norm)
        
        logging.info(f"Faiss索引构建完成")
        logging.info(f"标题索引大小: {self.title_index.ntotal}")
        logging.info(f"摘要索引大小: {self.abstract_index.ntotal}")
    
    def _search_with_faiss(self, query_embedding: np.ndarray, index, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """使用Faiss进行搜索"""
        if not FAISS_AVAILABLE or index is None:
            # 回退到numpy实现
            if index == self.title_index:
                similarities = cosine_similarity_batch_numpy(query_embedding, self.title_embeddings)
            else:
                similarities = cosine_similarity_batch_numpy(query_embedding, self.abstract_embeddings)
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            top_similarities = similarities[top_indices]
            return top_similarities, top_indices
        
        # 归一化查询向量
        query_norm = query_embedding.copy().reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_norm)
        
        # 搜索
        similarities, indices = index.search(query_norm, top_k)
        
        return similarities[0], indices[0]
    
    def search_by_title(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """基于标题嵌入搜索"""
        similarities, indices = self._search_with_faiss(query_embedding, self.title_index, top_k)
        
        results = []
        for i, (idx, similarity) in enumerate(zip(indices, similarities)):
            paper_id = self.paper_ids[idx].decode('utf-8') if isinstance(self.paper_ids[idx], bytes) else str(self.paper_ids[idx])
            results.append((paper_id, float(similarity)))
        
        return results
    
    def search_by_abstract(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """基于摘要嵌入搜索"""
        similarities, indices = self._search_with_faiss(query_embedding, self.abstract_index, top_k)
        
        results = []
        for i, (idx, similarity) in enumerate(zip(indices, similarities)):
            paper_id = self.paper_ids[idx].decode('utf-8') if isinstance(self.paper_ids[idx], bytes) else str(self.paper_ids[idx])
            results.append((paper_id, float(similarity)))
        
        return results
    
    def search_combined(self, query_embedding: np.ndarray, top_k: int = 10, 
                       title_weight: float = 0.3, abstract_weight: float = 0.7) -> List[Tuple[str, float]]:
        """组合标题和摘要嵌入进行搜索"""
        # 获取更多结果用于重新排序
        extended_k = min(top_k * 3, len(self.paper_ids))
        
        title_similarities, title_indices = self._search_with_faiss(query_embedding, self.title_index, extended_k)
        abstract_similarities, abstract_indices = self._search_with_faiss(query_embedding, self.abstract_index, extended_k)
        
        # 创建综合评分字典
        combined_scores = {}
        
        # 添加标题分数
        for idx, similarity in zip(title_indices, title_similarities):
            paper_id = self.paper_ids[idx].decode('utf-8') if isinstance(self.paper_ids[idx], bytes) else str(self.paper_ids[idx])
            combined_scores[paper_id] = title_weight * float(similarity)
        
        # 添加摘要分数
        for idx, similarity in zip(abstract_indices, abstract_similarities):
            paper_id = self.paper_ids[idx].decode('utf-8') if isinstance(self.paper_ids[idx], bytes) else str(self.paper_ids[idx])
            if paper_id in combined_scores:
                combined_scores[paper_id] += abstract_weight * float(similarity)
            else:
                combined_scores[paper_id] = abstract_weight * float(similarity)
        
        # 排序并返回top-k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return sorted_results
    
    def get_paper_info(self, paper_id: str) -> Optional[Dict]:
        """根据论文ID获取论文信息"""
        try:
            idx = np.where(self.paper_ids == paper_id.encode('utf-8'))[0]
            if len(idx) == 0:
                idx = np.where(self.paper_ids == paper_id)[0]
            
            if len(idx) > 0:
                return {
                    'id': paper_id,
                    'index': int(idx[0]),
                    'title_embedding_shape': self.title_embeddings[idx[0]].shape,
                    'abstract_embedding_shape': self.abstract_embeddings[idx[0]].shape
                }
        except Exception as e:
            logging.error(f"获取论文信息时出错: {str(e)}")
        
        return None
    
    def benchmark_search(self, query_embedding: np.ndarray, top_k: int = 10, num_runs: int = 10):
        """性能基准测试"""
        logging.info(f"开始性能基准测试 (运行 {num_runs} 次)...")
        
        # 测试标题搜索
        start_time = time.time()
        for _ in range(num_runs):
            self.search_by_title(query_embedding, top_k)
        title_time = (time.time() - start_time) / num_runs
        
        # 测试摘要搜索
        start_time = time.time()
        for _ in range(num_runs):
            self.search_by_abstract(query_embedding, top_k)
        abstract_time = (time.time() - start_time) / num_runs
        
        # 测试组合搜索
        start_time = time.time()
        for _ in range(num_runs):
            self.search_combined(query_embedding, top_k)
        combined_time = (time.time() - start_time) / num_runs
        
        logging.info(f"性能基准测试结果:")
        logging.info(f"  标题搜索平均耗时: {title_time*1000:.2f} ms")
        logging.info(f"  摘要搜索平均耗时: {abstract_time*1000:.2f} ms")
        logging.info(f"  组合搜索平均耗时: {combined_time*1000:.2f} ms")
        logging.info(f"  数据集大小: {len(self.paper_ids)} 篇论文")
        logging.info(f"  搜索吞吐量: {len(self.paper_ids)/(title_time*1000):.0f} 论文/ms")
    
    def close(self):
        """关闭H5文件和清理GPU资源"""
        if self.h5_file:
            self.h5_file.close()
            logging.info("H5文件已关闭")
        
        # 清理GPU资源
        if self.use_gpu and FAISS_AVAILABLE:
            try:
                # Faiss会自动管理GPU资源，但我们可以显式清理
                del self.title_index
                del self.abstract_index
                gc.collect()
                logging.info("GPU资源已清理")
            except Exception as e:
                logging.warning(f"清理GPU资源时出错: {str(e)}")


def interactive_search(searcher: GPUEmbeddingSearcher, api_url: str, model_name: str):
    """交互式搜索模式"""
    print("\n=" * 60)
    print("GPU加速交互式向量搜索模式")
    print(f"使用设备: {'GPU' if searcher.use_gpu else 'CPU'}")
    print("输入查询文本，系统将返回最相似的论文")
    print("输入 'benchmark' 进行性能测试")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    
    while True:
        try:
            query_text = input("\n请输入查询文本: ").strip()
            
            if query_text.lower() in ['quit', 'exit', 'q']:
                print("退出搜索模式")
                break
            
            if query_text.lower() == 'benchmark':
                # 性能测试
                test_embedding = call_vllm_service_sync("test query", api_url, 
                                                       config['prompts']['query_with_arxiv_paper_abstract'], 
                                                       model_name)
                searcher.benchmark_search(test_embedding)
                continue
            
            if not query_text:
                print("请输入有效的查询文本")
                continue
            
            # 选择搜索类型
            search_type = input("选择搜索类型 (1: 标题, 2: 摘要, 3: 组合) [默认: 3]: ").strip()
            if not search_type:
                search_type = "3"
            
            # 设置top-k
            top_k_input = input("返回结果数量 [默认: 10]: ").strip()
            top_k = int(top_k_input) if top_k_input.isdigit() else 10
            
            print(f"\n正在为查询 '{query_text}' 生成嵌入向量...")
            
            # 根据搜索类型选择prompt
            if search_type == "1":
                prompt = config['prompts']['query_with_arxiv_paper_title']
            elif search_type == "2":
                prompt = config['prompts']['query_with_arxiv_paper_abstract']
            else:
                prompt = config['prompts']['query_with_arxiv_paper_abstract']
            
            # 生成查询嵌入
            start_time = time.time()
            query_embedding = call_vllm_service_sync(query_text, api_url, prompt, model_name)
            embedding_time = time.time() - start_time
            
            print(f"嵌入生成完成，耗时: {embedding_time:.2f}秒")
            print("正在搜索相似论文...")
            
            # 执行搜索
            search_start = time.time()
            if search_type == "1":
                results = searcher.search_by_title(query_embedding, top_k)
                search_method = "标题"
            elif search_type == "2":
                results = searcher.search_by_abstract(query_embedding, top_k)
                search_method = "摘要"
            else:
                results = searcher.search_combined(query_embedding, top_k)
                search_method = "组合"
            
            search_time = time.time() - search_start
            
            # 显示结果
            print(f"\n搜索完成，耗时: {search_time*1000:.2f} ms")
            print(f"搜索方法: {search_method}")
            print(f"使用设备: {'GPU' if searcher.use_gpu else 'CPU'}")
            print(f"\nTop {len(results)} 相似论文:")
            print("-" * 80)
            
            for i, (paper_id, similarity) in enumerate(results, 1):
                print(f"{i:2d}. 论文ID: {paper_id}")
                print(f"    相似度: {similarity:.4f}")
                
                # 获取论文基本信息
                paper_info = searcher.get_paper_info(paper_id)
                if paper_info:
                    print(f"    索引: {paper_info['index']}")
                
                print()
        
        except KeyboardInterrupt:
            print("\n\n搜索被中断")
            break
        except Exception as e:
            print(f"搜索过程中出错: {str(e)}")
            logging.error(f"搜索错误: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="GPU加速的向量搜索工具")
    parser.add_argument("--h5_file", type=str, required=True,
                        help="H5嵌入文件路径")
    parser.add_argument("--metadata_file", type=str, default=None,
                        help="元数据文件路径（可选）")
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:8000/v1/embeddings",
                        help="vLLM OpenAI兼容Embedding API URL")
    parser.add_argument("--model_name", type=str, default="qwen3-embeddings-8b",
                        help="vLLM部署的模型名称")
    parser.add_argument("--query", type=str, default=None,
                        help="单次查询文本（如果不提供则进入交互模式）")
    parser.add_argument("--search_type", type=str, default="combined",
                        choices=["title", "abstract", "combined"],
                        help="搜索类型")
    parser.add_argument("--top_k", type=int, default=10,
                        help="返回结果数量")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                        help="使用GPU加速（默认启用）")
    parser.add_argument("--no_gpu", action="store_true", default=False,
                        help="禁用GPU，强制使用CPU")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU设备ID")
    parser.add_argument("--benchmark", action="store_true", default=False,
                        help="运行性能基准测试")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    
    args = parser.parse_args()
    
    # 处理GPU设置
    use_gpu = args.use_gpu and not args.no_gpu
    
    # 设置日志
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(log_level)
    
    # 输出配置信息
    logging.info("=" * 60)
    logging.info("GPU加速向量搜索工具启动")
    logging.info("=" * 60)
    for arg, value in sorted(vars(args).items()):
        logging.info(f"参数 {arg}: {value}")
    logging.info(f"Faiss可用: {FAISS_AVAILABLE}")
    if FAISS_AVAILABLE and use_gpu:
        try:
            import faiss
            gpu_count = faiss.get_num_gpus()
            logging.info(f"可用GPU数量: {gpu_count}")
        except:
            logging.warning("无法获取GPU信息")
    logging.info("=" * 60)
    
    try:
        # 测试vLLM服务连接
        logging.info("测试vLLM服务连接...")
        test_embedding = call_vllm_service_sync("test", args.api_url, None, args.model_name)
        logging.info(f"vLLM服务连接成功，嵌入维度: {len(test_embedding)}")
        
        # 初始化GPU搜索器
        logging.info("初始化GPU搜索器...")
        searcher = GPUEmbeddingSearcher(args.h5_file, args.metadata_file, use_gpu, args.gpu_id)
        
        if args.benchmark:
            # 性能基准测试
            logging.info("运行性能基准测试...")
            searcher.benchmark_search(test_embedding)
        
        if args.query:
            # 单次查询模式
            logging.info(f"执行单次查询: {args.query}")
            
            # 选择prompt
            if args.search_type == "title":
                prompt = config['prompts']['query_with_arxiv_paper_title']
            elif args.search_type == "abstract":
                prompt = config['prompts']['query_with_arxiv_paper_abstract']
            else:
                prompt = config['prompts']['query_with_arxiv_paper_abstract']
            
            # 生成查询嵌入
            start_time = time.time()
            query_embedding = call_vllm_service_sync(args.query, args.api_url, prompt, args.model_name)
            embedding_time = time.time() - start_time
            
            # 执行搜索
            search_start = time.time()
            if args.search_type == "title":
                results = searcher.search_by_title(query_embedding, args.top_k)
            elif args.search_type == "abstract":
                results = searcher.search_by_abstract(query_embedding, args.top_k)
            else:
                results = searcher.search_combined(query_embedding, args.top_k)
            search_time = time.time() - search_start
            
            # 输出结果
            print(f"\n嵌入生成耗时: {embedding_time:.2f}秒")
            print(f"搜索耗时: {search_time*1000:.2f} ms")
            print(f"使用设备: {'GPU' if searcher.use_gpu else 'CPU'}")
            print(f"\nTop {len(results)} 相似论文:")
            for i, (paper_id, similarity) in enumerate(results, 1):
                print(f"{i:2d}. 论文ID: {paper_id}, 相似度: {similarity:.4f}")
        
        elif not args.benchmark:
            # 交互式模式
            interactive_search(searcher, args.api_url, args.model_name)
        
        # 清理资源
        searcher.close()
        
    except Exception as e:
        logging.error(f"程序执行失败: {str(e)}")
        raise
    
    logging.info("程序执行完成")


if __name__ == "__main__":
    main()