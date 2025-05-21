#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
探索arXiv论文嵌入向量H5文件
提供简单的命令行工具查看H5文件内容和统计信息
"""

import os
import json
import argparse
import logging
import h5py
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_embeddings_h5(h5_file_path):
    """加载H5文件中的嵌入向量和论文ID"""
    with h5py.File(h5_file_path, 'r') as h5f:
        # 打印H5文件的基本结构
        logging.info("H5文件结构:")
        for key in h5f.keys():
            dataset = h5f[key]
            if isinstance(dataset, h5py.Dataset):
                shape_str = f"形状: {dataset.shape}, 类型: {dataset.dtype}"
            else:
                shape_str = "组"
            logging.info(f"  - {key}: {shape_str}")
        
        # 加载ID
        paper_ids = h5f['paper_ids'][:]
        # 将字节字符串转换为普通字符串
        if isinstance(paper_ids[0], bytes):
            paper_ids = [id.decode('utf-8') for id in paper_ids]
        
        # 加载嵌入向量
        title_embeddings = h5f['title_embeddings'][:]
        abstract_embeddings = h5f['abstract_embeddings'][:]
        
        logging.info(f"加载了 {len(paper_ids)} 篇论文的嵌入向量")
        logging.info(f"标题嵌入维度: {title_embeddings.shape}")
        logging.info(f"摘要嵌入维度: {abstract_embeddings.shape}")
    
    return paper_ids, title_embeddings, abstract_embeddings


def load_metadata(metadata_file_path):
    """加载元数据文件"""
    with open(metadata_file_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    id_to_index = metadata.get('id_to_index', {})
    papers = metadata.get('papers', [])
    embedding_info = metadata.get('embedding_info', {})
    
    logging.info(f"加载了 {len(papers)} 篇论文的元数据")
    if embedding_info:
        logging.info(f"嵌入信息: {embedding_info}")
    
    return papers, id_to_index


def display_embedding_statistics(embeddings, name="嵌入向量"):
    """显示嵌入向量的统计信息"""
    # 计算基本统计量
    means = np.mean(embeddings, axis=0)
    stds = np.std(embeddings, axis=0)
    mins = np.min(embeddings, axis=0)
    maxs = np.max(embeddings, axis=0)
    norms = np.linalg.norm(embeddings, axis=1)
    
    print(f"\n{name}统计信息:")
    print(f"  - 形状: {embeddings.shape}")
    print(f"  - 均值范围: [{np.min(means):.4f}, {np.max(means):.4f}]")
    print(f"  - 标准差范围: [{np.min(stds):.4f}, {np.max(stds):.4f}]")
    print(f"  - 最小值范围: [{np.min(mins):.4f}, {np.max(mins):.4f}]")
    print(f"  - 最大值范围: [{np.min(maxs):.4f}, {np.max(maxs):.4f}]")
    print(f"  - 向量范数: 均值={np.mean(norms):.4f}, 标准差={np.std(norms):.4f}")
    print(f"  - 范数范围: [{np.min(norms):.4f}, {np.max(norms):.4f}]")


def plot_embedding_pca(title_embeddings, abstract_embeddings, num_samples=100, output_file=None):
    """使用PCA绘制嵌入向量的2D投影"""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        logging.error("请安装scikit-learn以使用PCA功能: pip install scikit-learn")
        return
    
    # 随机选择样本
    if num_samples > title_embeddings.shape[0]:
        num_samples = title_embeddings.shape[0]
    
    indices = np.random.choice(title_embeddings.shape[0], num_samples, replace=False)
    title_samples = title_embeddings[indices]
    abstract_samples = abstract_embeddings[indices]
    
    # 应用PCA
    pca = PCA(n_components=2)
    title_pca = pca.fit_transform(title_samples)
    abstract_pca = pca.fit_transform(abstract_samples)
    
    # 绘图
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(title_pca[:, 0], title_pca[:, 1], alpha=0.5)
    plt.title('标题嵌入向量PCA')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    
    plt.subplot(1, 2, 2)
    plt.scatter(abstract_pca[:, 0], abstract_pca[:, 1], alpha=0.5)
    plt.title('摘要嵌入向量PCA')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        logging.info(f"PCA可视化已保存到: {output_file}")
    else:
        plt.show()


def display_paper_info(paper_idx, paper_id, paper, title_embedding, abstract_embedding):
    """显示单篇论文的详细信息"""
    print("\n" + "="*80)
    print(f"论文 #{paper_idx}: {paper_id}")
    print("="*80)
    
    print(f"标题: {paper['title']}")
    print(f"摘要: {paper['abstract'][:200]}...")  # 只显示部分摘要
    print(f"作者: {paper.get('authors', 'N/A')}")
    print(f"分类: {paper.get('categories', 'N/A')}")
    print(f"DOI: {paper.get('doi', 'N/A')}")
    print(f"更新日期: {paper.get('update_date', 'N/A')}")
    
    print("\n标题嵌入向量:")
    display_vector_preview(title_embedding)
    
    print("\n摘要嵌入向量:")
    display_vector_preview(abstract_embedding)
    
    print("\n标题嵌入向量统计:")
    print(f"  - 范数: {np.linalg.norm(title_embedding):.4f}")
    print(f"  - 均值: {np.mean(title_embedding):.4f}")
    print(f"  - 标准差: {np.std(title_embedding):.4f}")
    print(f"  - 最小值: {np.min(title_embedding):.4f}")
    print(f"  - 最大值: {np.max(title_embedding):.4f}")
    
    print("\n摘要嵌入向量统计:")
    print(f"  - 范数: {np.linalg.norm(abstract_embedding):.4f}")
    print(f"  - 均值: {np.mean(abstract_embedding):.4f}")
    print(f"  - 标准差: {np.std(abstract_embedding):.4f}")
    print(f"  - 最小值: {np.min(abstract_embedding):.4f}")
    print(f"  - 最大值: {np.max(abstract_embedding):.4f}")
    
    # 计算标题和摘要嵌入向量的余弦相似度
    cos_sim = np.dot(title_embedding, abstract_embedding) / (
        np.linalg.norm(title_embedding) * np.linalg.norm(abstract_embedding)
    )
    print(f"\n标题和摘要嵌入向量的余弦相似度: {cos_sim:.4f}")


def display_vector_preview(vector, max_dims=10):
    """显示向量的前几个维度"""
    preview = vector[:max_dims]
    preview_str = ", ".join([f"{x:.4f}" for x in preview])
    print(f"[{preview_str}, ... ]（共{len(vector)}维）")


def explore_embeddings(h5_file_path, metadata_file_path, paper_id=None, paper_index=None, sample_size=5, plot_pca=False):
    """探索嵌入向量文件"""
    # 加载嵌入向量和ID
    paper_ids, title_embeddings, abstract_embeddings = load_embeddings_h5(h5_file_path)
    
    # 加载元数据
    papers, id_to_index = load_metadata(metadata_file_path)
    
    # 显示基本统计信息
    display_embedding_statistics(title_embeddings, "标题嵌入向量")
    display_embedding_statistics(abstract_embeddings, "摘要嵌入向量")
    
    # 检查ID映射一致性
    h5_id_set = set(paper_ids)
    meta_id_set = set(id_to_index.keys())
    common_ids = h5_id_set.intersection(meta_id_set)
    
    logging.info(f"H5文件中的ID数量: {len(h5_id_set)}")
    logging.info(f"元数据中的ID数量: {len(meta_id_set)}")
    logging.info(f"共同ID数量: {len(common_ids)}")
    
    # 显示特定论文信息
    if paper_id:
        if paper_id in id_to_index:
            meta_idx = id_to_index[paper_id]
            paper = papers[meta_idx]
            
            # 查找在H5文件中的索引
            try:
                h5_idx = list(paper_ids).index(paper_id)
                title_emb = title_embeddings[h5_idx]
                abstract_emb = abstract_embeddings[h5_idx]
                
                display_paper_info(h5_idx, paper_id, paper, title_emb, abstract_emb)
            except ValueError:
                logging.error(f"在H5文件中找不到ID: {paper_id}")
        else:
            logging.error(f"在元数据中找不到ID: {paper_id}")
    
    elif paper_index is not None:
        if 0 <= paper_index < len(paper_ids):
            paper_id = paper_ids[paper_index]
            title_emb = title_embeddings[paper_index]
            abstract_emb = abstract_embeddings[paper_index]
            
            if paper_id in id_to_index:
                meta_idx = id_to_index[paper_id]
                paper = papers[meta_idx]
                display_paper_info(paper_index, paper_id, paper, title_emb, abstract_emb)
            else:
                logging.error(f"在元数据中找不到对应的论文: {paper_id}")
        else:
            logging.error(f"索引超出范围: {paper_index}，文件包含 {len(paper_ids)} 篇论文")
    
    # 显示随机样本信息
    else:
        if sample_size > len(paper_ids):
            sample_size = len(paper_ids)
            
        indices = np.random.choice(len(paper_ids), sample_size, replace=False)
        
        print("\n随机样本预览:")
        table_data = []
        headers = ["索引", "论文ID", "标题", "向量范数(标题/摘要)"]
        
        for idx in indices:
            paper_id = paper_ids[idx]
            title_norm = np.linalg.norm(title_embeddings[idx])
            abstract_norm = np.linalg.norm(abstract_embeddings[idx])
            
            if paper_id in id_to_index:
                meta_idx = id_to_index[paper_id]
                title = papers[meta_idx].get('title', 'N/A')
                # 截断标题，避免表格太宽
                if len(title) > 50:
                    title = title[:47] + "..."
            else:
                title = "未找到元数据"
                
            table_data.append([
                idx, 
                paper_id, 
                title, 
                f"{title_norm:.4f}/{abstract_norm:.4f}"
            ])
            
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # 绘制PCA可视化
    if plot_pca:
        plot_embedding_pca(title_embeddings, abstract_embeddings, output_file="embeddings_pca.png")


def main():
    parser = argparse.ArgumentParser(description="探索arXiv论文嵌入向量H5文件")
    parser.add_argument("--h5_file", type=str, required=True,
                        help="H5文件路径，包含嵌入向量")
    parser.add_argument("--metadata_file", type=str, required=True,
                        help="元数据JSON文件路径")
    parser.add_argument("--paper_id", type=str,
                        help="要查看的特定论文ID")
    parser.add_argument("--index", type=int, 
                        help="要查看的特定论文索引")
    parser.add_argument("--samples", type=int, default=5,
                        help="要显示的随机样本数量")
    parser.add_argument("--plot_pca", action="store_true",
                        help="绘制嵌入向量的PCA可视化")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 探索嵌入向量
    explore_embeddings(
        args.h5_file,
        args.metadata_file,
        args.paper_id,
        args.index,
        args.samples,
        args.plot_pca
    )


if __name__ == "__main__":
    main() 