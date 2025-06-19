#!/usr/bin/env python3
"""
基于H5文件的向量搜索工具
加载预生成的嵌入向量，对输入文本进行编码并搜索最相似的论文
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
                embedding = np.array(resp_json["data"][0]["embedding"], dtype=np.float16)
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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cosine_similarity_batch(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """批量计算查询向量与所有向量的余弦相似度"""
    # 归一化查询向量
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    
    # 归一化所有嵌入向量
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # 计算余弦相似度
    similarities = np.dot(embeddings_norm, query_norm)
    
    return similarities


class EmbeddingSearcher:
    """嵌入向量搜索器"""
    
    def __init__(self, h5_file_path: str, metadata_file_path: Optional[str] = None):
        self.h5_file_path = h5_file_path
        self.metadata_file_path = metadata_file_path
        self.h5_file = None
        self.metadata = None
        self.paper_ids = None
        self.title_embeddings = None
        self.abstract_embeddings = None
        self.embedding_dim = None
        
        self._load_data()
    
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
        self.title_embeddings = self.h5_file['title_embeddings'][:]
        self.abstract_embeddings = self.h5_file['abstract_embeddings'][:]
        
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
    
    def search_by_title(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """基于标题嵌入搜索"""
        similarities = cosine_similarity_batch(query_embedding, self.title_embeddings)
        
        # 获取top-k索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            paper_id = self.paper_ids[idx].decode('utf-8') if isinstance(self.paper_ids[idx], bytes) else str(self.paper_ids[idx])
            similarity = float(similarities[idx])
            results.append((paper_id, similarity))
        
        return results
    
    def search_by_abstract(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """基于摘要嵌入搜索"""
        similarities = cosine_similarity_batch(query_embedding, self.abstract_embeddings)
        
        # 获取top-k索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            paper_id = self.paper_ids[idx].decode('utf-8') if isinstance(self.paper_ids[idx], bytes) else str(self.paper_ids[idx])
            similarity = float(similarities[idx])
            results.append((paper_id, similarity))
        
        return results
    
    def search_combined(self, query_embedding: np.ndarray, top_k: int = 10, 
                       title_weight: float = 0.3, abstract_weight: float = 0.7) -> List[Tuple[str, float]]:
        """组合标题和摘要嵌入进行搜索"""
        title_similarities = cosine_similarity_batch(query_embedding, self.title_embeddings)
        abstract_similarities = cosine_similarity_batch(query_embedding, self.abstract_embeddings)
        
        # 加权组合
        combined_similarities = title_weight * title_similarities + abstract_weight * abstract_similarities
        
        # 获取top-k索引
        top_indices = np.argsort(combined_similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            paper_id = self.paper_ids[idx].decode('utf-8') if isinstance(self.paper_ids[idx], bytes) else str(self.paper_ids[idx])
            similarity = float(combined_similarities[idx])
            results.append((paper_id, similarity))
        
        return results
    
    def get_paper_info(self, paper_id: str) -> Optional[Dict]:
        """根据论文ID获取论文信息（需要原始数据文件）"""
        # 这里只返回基本信息，如果需要完整信息需要加载原始数据文件
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
    
    def close(self):
        """关闭H5文件"""
        if self.h5_file:
            self.h5_file.close()
            logging.info("H5文件已关闭")


def interactive_search(searcher: EmbeddingSearcher, api_url: str, model_name: str):
    """交互式搜索模式"""
    print("\n=" * 60)
    print("交互式向量搜索模式")
    print("输入查询文本，系统将返回最相似的论文")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    
    while True:
        try:
            query_text = input("\n请输入查询文本: ").strip()
            
            if query_text.lower() in ['quit', 'exit', 'q']:
                print("退出搜索模式")
                break
            
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
                prompt = config['prompts']['query_with_arxiv_paper_abstract']  # 默认使用摘要prompt
            
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
            print(f"\n搜索完成，耗时: {search_time:.3f}秒")
            print(f"搜索方法: {search_method}")
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
    parser = argparse.ArgumentParser(description="基于H5文件的向量搜索工具")
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
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(log_level)
    
    # 输出配置信息
    logging.info("=" * 60)
    logging.info("向量搜索工具启动")
    logging.info("=" * 60)
    for arg, value in sorted(vars(args).items()):
        logging.info(f"参数 {arg}: {value}")
    logging.info("=" * 60)
    
    try:
        # 测试vLLM服务连接
        logging.info("测试vLLM服务连接...")
        test_embedding = call_vllm_service_sync("test", args.api_url, None, args.model_name)
        logging.info(f"vLLM服务连接成功，嵌入维度: {len(test_embedding)}")
        
        # 初始化搜索器
        logging.info("初始化搜索器...")
        searcher = EmbeddingSearcher(args.h5_file, args.metadata_file)
        
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
            query_embedding = call_vllm_service_sync(args.query, args.api_url, prompt, args.model_name)
            
            # 执行搜索
            if args.search_type == "title":
                results = searcher.search_by_title(query_embedding, args.top_k)
            elif args.search_type == "abstract":
                results = searcher.search_by_abstract(query_embedding, args.top_k)
            else:
                results = searcher.search_combined(query_embedding, args.top_k)
            
            # 输出结果
            print(f"\nTop {len(results)} 相似论文:")
            for i, (paper_id, similarity) in enumerate(results, 1):
                print(f"{i:2d}. 论文ID: {paper_id}, 相似度: {similarity:.4f}")
        
        else:
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