#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用语义嵌入搜索arXiv论文
"""

import os
import argparse
import json
import logging
from sentence_transformers import SentenceTransformer


def search_papers_by_vector(
    query_text,
    embedding_file,
    metadata_file,
    model,
    top_k=10,
    use_title=True,
    use_abstract=True
):
    """
    通过语义搜索检索相关论文
    
    参数:
        query_text (str): 查询文本
        embedding_file (str): 嵌入向量文件路径 (h5格式)
        metadata_file (str): 元数据文件路径 (json格式)
        model: 嵌入模型
        top_k (int): 返回的最相关论文数量
        use_title (bool): 是否使用标题嵌入向量搜索
        use_abstract (bool): 是否使用摘要嵌入向量搜索
    
    返回:
        list: 最相关论文的信息列表
    """
    import h5py
    import numpy as np
    from scipy.spatial.distance import cdist
    
    # 生成查询文本的嵌入向量
    query_embedding = model.encode([query_text])[0]
    
    # 读取元数据
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 加载嵌入向量
    with h5py.File(embedding_file, 'r') as f:
        paper_ids = f['paper_ids'][:]
        
        # 根据选择使用标题或摘要嵌入进行搜索
        if use_title and not use_abstract:
            embeddings = f['title_embeddings'][:]
        elif use_abstract and not use_title:
            embeddings = f['abstract_embeddings'][:]
        else:
            # 同时使用标题和摘要（计算平均值）
            title_embeddings = f['title_embeddings'][:]
            abstract_embeddings = f['abstract_embeddings'][:]
            embeddings = (title_embeddings + abstract_embeddings) / 2
    
    # 计算余弦距离
    distances = cdist([query_embedding], embeddings, metric='cosine')[0]
    
    # 获取最相似的论文索引
    top_indices = np.argsort(distances)[:top_k]
    
    # 准备结果
    results = []
    id_to_index = metadata.get('id_to_index', {})
    
    for idx in top_indices:
        paper_id = paper_ids[idx]
        
        # 通过paper_id获取元数据
        if id_to_index:
            # 使用映射获取元数据索引
            metadata_idx = id_to_index.get(paper_id)
            if metadata_idx is not None and metadata_idx < len(metadata['papers']):
                paper_info = metadata['papers'][metadata_idx]
            else:
                # 如果找不到映射，则尝试线性搜索
                paper_info = next((p for p in metadata['papers'] if p['id'] == paper_id), {})
        else:
            # 如果没有映射，则使用线性搜索
            paper_info = next((p for p in metadata['papers'] if p['id'] == paper_id), {})
        
        # 添加相似度分数
        similarity = 1 - distances[idx]  # 转换为相似度
        paper_info['similarity'] = float(similarity)
        
        results.append(paper_info)
    
    return results

def search_papers_by_vector_numpy(
    query_text,
    embedding_files,
    metadata_file,
    model,
    top_k=10,
    use_title=True,
    use_abstract=True
):
    """
    通过语义搜索检索相关论文（针对numpy格式存储）
    
    参数:
        query_text (str): 查询文本
        embedding_files (tuple): 嵌入向量文件路径元组 (title_base, abstract_base)
        metadata_file (str): 元数据文件路径 (json格式)
        model: 嵌入模型
        top_k (int): 返回的最相关论文数量
        use_title (bool): 是否使用标题嵌入向量搜索
        use_abstract (bool): 是否使用摘要嵌入向量搜索
    
    返回:
        list: 最相关论文的信息列表
    """
    import numpy as np
    from scipy.spatial.distance import cdist
    from pathlib import Path
    
    # 生成查询文本的嵌入向量
    query_embedding = model.encode([query_text])[0]
    
    # 读取元数据
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    title_base, abstract_base = embedding_files
    
    # 获取所有部分文件
    title_parts = sorted(list(Path(os.path.dirname(title_base)).glob(f"{os.path.basename(title_base)}_part*.npz")))
    abstract_parts = sorted(list(Path(os.path.dirname(abstract_base)).glob(f"{os.path.basename(abstract_base)}_part*.npz")))
    
    # 记录所有文件的结果
    all_distances = []
    all_ids = []
    
    # 对每个部分文件进行搜索
    for i in range(len(title_parts)):
        title_part = title_parts[i]
        abstract_part = abstract_parts[i]
        
        # 加载嵌入向量
        title_data = np.load(title_part)
        abstract_data = np.load(abstract_part)
        
        # 确保ID一致
        if not np.array_equal(title_data['ids'], abstract_data['ids']):
            logging.warning(f"文件 {title_part} 和 {abstract_part} 的ID不匹配")
            continue
        
        # 根据选择使用标题或摘要嵌入进行搜索
        if use_title and not use_abstract:
            embeddings = title_data['embeddings']
        elif use_abstract and not use_title:
            embeddings = abstract_data['embeddings']
        else:
            # 同时使用标题和摘要（计算平均值）
            embeddings = (title_data['embeddings'] + abstract_data['embeddings']) / 2
        
        # 计算余弦距离
        distances = cdist([query_embedding], embeddings, metric='cosine')[0]
        
        # 保存当前部分的距离和ID
        all_distances.append(distances)
        all_ids.append(title_data['ids'])
    
    # 合并所有部分的结果
    if all_distances:
        all_distances = np.concatenate(all_distances)
        all_ids = np.concatenate(all_ids)
        
        # 获取最相似的论文索引
        top_indices = np.argsort(all_distances)[:top_k]
        
        # 准备结果
        results = []
        id_to_index = metadata.get('id_to_index', {})
        
        for idx in top_indices:
            paper_id = all_ids[idx]
            
            # 通过paper_id获取元数据
            if id_to_index:
                # 使用映射获取元数据索引
                metadata_idx = id_to_index.get(paper_id)
                if metadata_idx is not None and metadata_idx < len(metadata['papers']):
                    paper_info = metadata['papers'][metadata_idx]
                else:
                    # 如果找不到映射，则尝试线性搜索
                    paper_info = next((p for p in metadata['papers'] if p['id'] == paper_id), {})
            else:
                # 如果没有映射，则使用线性搜索
                paper_info = next((p for p in metadata['papers'] if p['id'] == paper_id), {})
            
            # 添加相似度分数
            similarity = 1 - all_distances[idx]  # 转换为相似度
            paper_info['similarity'] = float(similarity)
            
            results.append(paper_info)
        
        return results
    else:
        return []

def setup_logging():
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description="语义搜索arXiv论文")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="搜索查询文本"
    )
    parser.add_argument(
        "--embedding_file",
        type=str,
        required=True,
        help="嵌入向量文件路径（h5文件或numpy基础名）"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        required=True,
        help="元数据文件路径"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/e5-mistral-7b-instruct",
        help="模型路径"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="返回的最相关结果数量"
    )
    parser.add_argument(
        "--use_title",
        action="store_true",
        default=True,
        help="使用标题嵌入进行搜索"
    )
    parser.add_argument(
        "--use_abstract",
        action="store_true",
        default=True,
        help="使用摘要嵌入进行搜索"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["h5", "numpy"],
        default="h5",
        help="嵌入向量存储格式"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    logging.info(f"加载模型: {args.model_path}")
    model = SentenceTransformer(args.model_path)
    
    logging.info(f"搜索查询: '{args.query}'")
    
    if args.format == "h5":
        results = search_papers_by_vector(
            args.query,
            args.embedding_file,
            args.metadata_file,
            model,
            top_k=args.top_k,
            use_title=args.use_title,
            use_abstract=args.use_abstract
        )
    else:  # numpy格式
        # 假设embedding_file是基础文件名，需要拆分为标题和摘要基础名
        dir_path = os.path.dirname(args.embedding_file)
        base_name = os.path.basename(args.embedding_file)
        title_base = os.path.join(dir_path, f"{base_name}_title")
        abstract_base = os.path.join(dir_path, f"{base_name}_abstract")
        
        results = search_papers_by_vector_numpy(
            args.query,
            (title_base, abstract_base),
            args.metadata_file,
            model,
            top_k=args.top_k,
            use_title=args.use_title,
            use_abstract=args.use_abstract
        )
    
    logging.info(f"找到 {len(results)} 个相关结果")
    
    # 打印结果
    for i, paper in enumerate(results):
        print(f"\n--- 结果 {i+1} (相似度: {paper['similarity']:.4f}) ---")
        print(f"ID: {paper['id']}")
        print(f"标题: {paper['title']}")
        print(f"摘要: {paper['abstract'][:300]}..." if len(paper['abstract']) > 300 else f"摘要: {paper['abstract']}")
        print(f"作者: {paper['authors']}")
        print(f"类别: {paper['categories']}")
        if paper.get('doi'):
            print(f"DOI: {paper['doi']}")
        if paper.get('journal_ref'):
            print(f"期刊引用: {paper['journal_ref']}")
    
    # 将结果保存到JSON文件
    output_file = f"search_results_{args.query.replace(' ', '_')[:20]}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(
            {
                "query": args.query,
                "results": results
            },
            f,
            ensure_ascii=False,
            indent=2
        )
    
    logging.info(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    main() 