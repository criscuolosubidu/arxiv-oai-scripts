#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用Qdrant进行语义搜索的示例脚本
"""

import argparse
import logging
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from qdrant_client import QdrantClient
except ImportError:
    print("请安装 qdrant-client: pip install qdrant-client")
    exit(1)


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def search_papers(
    query_text: str,
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "arxiv_oai",
    model_path: str = "models/e5-mistral-7b-instruct",
    vector_name: str = "title",
    top_k: int = 10,
    score_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    使用Qdrant搜索相似论文
    
    参数:
        query_text: 查询文本
        qdrant_url: Qdrant服务URL
        collection_name: 集合名称
        model_path: 嵌入模型路径
        vector_name: 使用的向量名称 ("title" 或 "abstract")
        top_k: 返回结果数量
        score_threshold: 相似度阈值
    
    返回:
        搜索结果列表
    """
    
    # 加载模型
    logging.info(f"加载模型: {model_path}")
    model = SentenceTransformer(model_path)
    
    # 生成查询向量
    logging.info(f"生成查询向量: '{query_text}'")
    query_vector = model.encode([query_text])[0]
    
    # 对查询向量进行L2归一化，与Qdrant的余弦相似度计算保持一致
    query_vector = query_vector / np.linalg.norm(query_vector)
    query_vector = query_vector.tolist()
    
    # 连接Qdrant
    logging.info(f"连接Qdrant: {qdrant_url}")
    client = QdrantClient(url=qdrant_url)
    
    # 搜索
    logging.info(f"在集合 '{collection_name}' 中搜索...")
    results = client.search(
        collection_name=collection_name,
        query_vector=(vector_name, query_vector),
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True
    )
    
    # 格式化结果
    formatted_results = []
    for result in results:
        formatted_results.append({
            "id": result.id,
            "score": result.score,
            "paper_id": result.payload.get("paper_id", ""),
            "title": result.payload.get("title", ""),
            "abstract": result.payload.get("abstract", ""),
            "authors": result.payload.get("authors", ""),
            "categories": result.payload.get("categories", ""),
            "doi": result.payload.get("doi", ""),
            "journal_ref": result.payload.get("journal_ref", "")
        })
    
    return formatted_results


def main():
    parser = argparse.ArgumentParser(description="使用Qdrant进行语义搜索")
    parser.add_argument("--query", type=str, required=True,
                        help="搜索查询文本")
    parser.add_argument("--qdrant_url", type=str, default="http://localhost:6333",
                        help="Qdrant服务URL")
    parser.add_argument("--collection_name", type=str, default="arxiv_oai",
                        help="Qdrant集合名称")
    parser.add_argument("--model_path", type=str, default="models/e5-mistral-7b-instruct",
                        help="嵌入模型路径")
    parser.add_argument("--vector_name", type=str, default="title",
                        choices=["title", "abstract"],
                        help="使用的向量类型")
    parser.add_argument("--top_k", type=int, default=10,
                        help="返回结果数量")
    parser.add_argument("--score_threshold", type=float, default=0.7,
                        help="相似度阈值")
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        # 执行搜索
        results = search_papers(
            query_text=args.query,
            qdrant_url=args.qdrant_url,
            collection_name=args.collection_name,
            model_path=args.model_path,
            vector_name=args.vector_name,
            top_k=args.top_k,
            score_threshold=args.score_threshold
        )
        
        # 显示结果
        logging.info(f"找到 {len(results)} 个相关结果")
        
        for i, paper in enumerate(results, 1):
            print(f"\n--- 结果 {i} (相似度: {paper['score']:.4f}) ---")
            print(f"论文ID: {paper['paper_id']}")
            print(f"标题: {paper['title']}")
            print(f"摘要: {paper['abstract'][:300]}..." if len(paper['abstract']) > 300 else f"摘要: {paper['abstract']}")
            print(f"作者: {paper['authors']}")
            print(f"类别: {paper['categories']}")
            if paper['doi']:
                print(f"DOI: {paper['doi']}")
            if paper['journal_ref']:
                print(f"期刊引用: {paper['journal_ref']}")
        
        if not results:
            print("没有找到符合条件的结果")
            
    except Exception as e:
        logging.error(f"搜索过程中发生错误: {str(e)}")


if __name__ == "__main__":
    main()