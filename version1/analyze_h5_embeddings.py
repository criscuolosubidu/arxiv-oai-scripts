#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
arXiv嵌入向量H5文件处理工具
功能包括：文件分析、统计信息、质量检查、搜索等
"""

import h5py
import numpy as np
import json
import os
import argparse
import logging
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def setup_logger():
    """设置日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'h5_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class H5EmbeddingAnalyzer:
    """H5嵌入向量文件分析器"""
    
    def __init__(self, h5_file_path, metadata_file_path=None):
        self.h5_file_path = h5_file_path
        self.metadata_file_path = metadata_file_path
        self.logger = logging.getLogger(__name__)
        
        # 加载元数据
        self.metadata = None
        if metadata_file_path and os.path.exists(metadata_file_path):
            with open(metadata_file_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        
        # 打开H5文件
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.paper_ids = self.h5_file['paper_ids'][:]
        self.title_embeddings = self.h5_file['title_embeddings']
        self.abstract_embeddings = self.h5_file['abstract_embeddings']
        
        self.num_papers = len(self.paper_ids)
        self.embedding_dim = self.title_embeddings.shape[1]
        
        self.logger.info(f"加载H5文件: {h5_file_path}")
        self.logger.info(f"论文数量: {self.num_papers:,}")
        self.logger.info(f"嵌入维度: {self.embedding_dim}")
    
    def get_basic_stats(self):
        """获取基本统计信息"""
        file_size_gb = os.path.getsize(self.h5_file_path) / (1024**3)
        
        stats = {
            '文件路径': self.h5_file_path,
            '文件大小': f"{file_size_gb:.2f} GB",
            '论文数量': f"{self.num_papers:,}",
            '嵌入维度': self.embedding_dim,
            '数据类型': str(self.title_embeddings.dtype),
            '压缩方式': self.title_embeddings.compression,
            '创建时间': self.metadata.get('embedding_info', {}).get('creation_date', '未知') if self.metadata else '未知'
        }
        
        return stats
    
    def analyze_embedding_quality(self, sample_size=1000):
        """分析嵌入向量质量"""
        self.logger.info("开始分析嵌入向量质量...")
        
        # 随机采样
        if self.num_papers > sample_size:
            indices = np.random.choice(self.num_papers, sample_size, replace=False)
        else:
            indices = np.arange(self.num_papers)
        
        # 加载采样数据
        title_sample = self.title_embeddings[indices]
        abstract_sample = self.abstract_embeddings[indices]
        
        # 计算统计指标
        title_stats = {
            '均值': np.mean(title_sample),
            '标准差': np.std(title_sample),
            '最小值': np.min(title_sample),
            '最大值': np.max(title_sample),
            '零值比例': np.mean(title_sample == 0),
            '模长均值': np.mean(np.linalg.norm(title_sample, axis=1)),
            'NaN数量': np.sum(np.isnan(title_sample))
        }
        
        abstract_stats = {
            '均值': np.mean(abstract_sample),
            '标准差': np.std(abstract_sample),
            '最小值': np.min(abstract_sample),
            '最大值': np.max(abstract_sample),
            '零值比例': np.mean(abstract_sample == 0),
            '模长均值': np.mean(np.linalg.norm(abstract_sample, axis=1)),
            'NaN数量': np.sum(np.isnan(abstract_sample))
        }
        
        # 计算标题和摘要嵌入的相似性
        similarities = []
        for i in range(len(title_sample)):
            sim = cosine_similarity(
                title_sample[i:i+1], 
                abstract_sample[i:i+1]
            )[0][0]
            similarities.append(sim)
        
        similarity_stats = {
            '平均相似度': np.mean(similarities),
            '相似度标准差': np.std(similarities),
            '最小相似度': np.min(similarities),
            '最大相似度': np.max(similarities)
        }
        
        return {
            '标题嵌入统计': title_stats,
            '摘要嵌入统计': abstract_stats,
            '标题-摘要相似性': similarity_stats,
            '采样大小': len(indices)
        }
    
    def find_similar_papers(self, query_id, top_k=10, use_title=True):
        """根据论文ID查找相似论文"""
        try:
            # 找到查询论文的索引
            query_idx = np.where(self.paper_ids == query_id.encode('utf-8'))[0]
            if len(query_idx) == 0:
                return None, f"论文ID '{query_id}' 未找到"
            
            query_idx = query_idx[0]
            
            # 获取查询向量
            if use_title:
                query_vector = self.title_embeddings[query_idx:query_idx+1]
                search_embeddings = self.title_embeddings
                search_type = "标题"
            else:
                query_vector = self.abstract_embeddings[query_idx:query_idx+1]
                search_embeddings = self.abstract_embeddings
                search_type = "摘要"
            
            # 计算相似度（分批处理以节省内存）
            batch_size = 10000
            similarities = []
            
            for i in tqdm(range(0, self.num_papers, batch_size), desc=f"计算{search_type}相似度"):
                end_idx = min(i + batch_size, self.num_papers)
                batch_embeddings = search_embeddings[i:end_idx]
                batch_sims = cosine_similarity(query_vector, batch_embeddings)[0]
                similarities.extend(batch_sims)
            
            similarities = np.array(similarities)
            
            # 获取最相似的论文
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    '排名': len(results) + 1,
                    '论文ID': self.paper_ids[idx].decode('utf-8'),
                    '相似度': f"{similarities[idx]:.4f}",
                    '索引': idx
                })
            
            return results, None
            
        except Exception as e:
            return None, f"搜索过程中出错: {str(e)}"
    
    def export_sample_data(self, output_dir, sample_size=100):
        """导出采样数据用于进一步分析"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 随机采样
        indices = np.random.choice(self.num_papers, min(sample_size, self.num_papers), replace=False)
        
        # 导出数据
        sample_ids = [self.paper_ids[i].decode('utf-8') for i in indices]
        sample_title_embeddings = self.title_embeddings[indices]
        sample_abstract_embeddings = self.abstract_embeddings[indices]
        
        # 保存为numpy格式
        np.save(os.path.join(output_dir, 'sample_ids.npy'), sample_ids)
        np.save(os.path.join(output_dir, 'sample_title_embeddings.npy'), sample_title_embeddings)
        np.save(os.path.join(output_dir, 'sample_abstract_embeddings.npy'), sample_abstract_embeddings)
        
        # 保存为CSV格式（仅ID和统计信息）
        df_data = []
        for i, idx in enumerate(indices):
            title_norm = np.linalg.norm(sample_title_embeddings[i])
            abstract_norm = np.linalg.norm(sample_abstract_embeddings[i])
            similarity = cosine_similarity(
                sample_title_embeddings[i:i+1], 
                sample_abstract_embeddings[i:i+1]
            )[0][0]
            
            df_data.append({
                'paper_id': sample_ids[i],
                'title_embedding_norm': title_norm,
                'abstract_embedding_norm': abstract_norm,
                'title_abstract_similarity': similarity
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(os.path.join(output_dir, 'sample_stats.csv'), index=False)
        
        self.logger.info(f"采样数据已导出到: {output_dir}")
        return output_dir
    
    def visualize_embeddings(self, output_dir, sample_size=1000):
        """可视化嵌入向量分布"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 随机采样
        if self.num_papers > sample_size:
            indices = np.random.choice(self.num_papers, sample_size, replace=False)
        else:
            indices = np.arange(self.num_papers)
        
        title_sample = self.title_embeddings[indices]
        abstract_sample = self.abstract_embeddings[indices]
        
        # 1. 嵌入向量模长分布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        title_norms = np.linalg.norm(title_sample, axis=1)
        abstract_norms = np.linalg.norm(abstract_sample, axis=1)
        
        ax1.hist(title_norms, bins=50, alpha=0.7, label='标题嵌入')
        ax1.set_xlabel('嵌入向量模长')
        ax1.set_ylabel('频次')
        ax1.set_title('标题嵌入向量模长分布')
        ax1.legend()
        
        ax2.hist(abstract_norms, bins=50, alpha=0.7, label='摘要嵌入', color='orange')
        ax2.set_xlabel('嵌入向量模长')
        ax2.set_ylabel('频次')
        ax2.set_title('摘要嵌入向量模长分布')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'embedding_norms_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 标题-摘要相似度分布
        similarities = []
        for i in range(len(title_sample)):
            sim = cosine_similarity(title_sample[i:i+1], abstract_sample[i:i+1])[0][0]
            similarities.append(sim)
        
        plt.figure(figsize=(10, 6))
        plt.hist(similarities, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('余弦相似度')
        plt.ylabel('频次')
        plt.title(f'标题-摘要嵌入相似度分布 (采样大小: {len(title_sample):,})')
        plt.axvline(np.mean(similarities), color='red', linestyle='--', label=f'平均值: {np.mean(similarities):.3f}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'title_abstract_similarity_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. PCA降维可视化
        if self.embedding_dim > 2:
            # 对标题和摘要嵌入分别进行PCA
            pca = PCA(n_components=2)
            title_pca = pca.fit_transform(title_sample)
            
            pca_abstract = PCA(n_components=2)
            abstract_pca = pca_abstract.fit_transform(abstract_sample)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.scatter(title_pca[:, 0], title_pca[:, 1], alpha=0.6, s=1)
            ax1.set_xlabel(f'PC1 (解释方差: {pca.explained_variance_ratio_[0]:.3f})')
            ax1.set_ylabel(f'PC2 (解释方差: {pca.explained_variance_ratio_[1]:.3f})')
            ax1.set_title('标题嵌入 PCA 可视化')
            
            ax2.scatter(abstract_pca[:, 0], abstract_pca[:, 1], alpha=0.6, s=1, color='orange')
            ax2.set_xlabel(f'PC1 (解释方差: {pca_abstract.explained_variance_ratio_[0]:.3f})')
            ax2.set_ylabel(f'PC2 (解释方差: {pca_abstract.explained_variance_ratio_[1]:.3f})')
            ax2.set_title('摘要嵌入 PCA 可视化')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'embeddings_pca_visualization.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"可视化图表已保存到: {output_dir}")
    
    def validate_integrity(self):
        """验证H5文件完整性"""
        issues = []
        
        # 检查数据形状一致性
        if len(self.paper_ids) != self.title_embeddings.shape[0]:
            issues.append(f"论文ID数量 ({len(self.paper_ids)}) 与标题嵌入数量 ({self.title_embeddings.shape[0]}) 不匹配")
        
        if len(self.paper_ids) != self.abstract_embeddings.shape[0]:
            issues.append(f"论文ID数量 ({len(self.paper_ids)}) 与摘要嵌入数量 ({self.abstract_embeddings.shape[0]}) 不匹配")
        
        if self.title_embeddings.shape[1] != self.abstract_embeddings.shape[1]:
            issues.append(f"标题嵌入维度 ({self.title_embeddings.shape[1]}) 与摘要嵌入维度 ({self.abstract_embeddings.shape[1]}) 不匹配")
        
        # 检查重复ID
        unique_ids = np.unique(self.paper_ids)
        if len(unique_ids) != len(self.paper_ids):
            issues.append(f"发现重复的论文ID，唯一ID数量: {len(unique_ids)}, 总ID数量: {len(self.paper_ids)}")
        
        # 检查NaN值
        title_nan_count = np.sum(np.isnan(self.title_embeddings[:1000]))  # 检查前1000条
        abstract_nan_count = np.sum(np.isnan(self.abstract_embeddings[:1000]))
        
        if title_nan_count > 0:
            issues.append(f"标题嵌入中发现 {title_nan_count} 个NaN值（前1000条中）")
        if abstract_nan_count > 0:
            issues.append(f"摘要嵌入中发现 {abstract_nan_count} 个NaN值（前1000条中）")
        
        return issues
    
    def close(self):
        """关闭H5文件"""
        self.h5_file.close()


def print_stats_table(stats_dict, title="统计信息"):
    """打印格式化的统计表格"""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    
    for key, value in stats_dict.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    print(f"  {sub_key}: {sub_value:.6f}")
                else:
                    print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="arXiv嵌入向量H5文件分析工具")
    parser.add_argument("--h5_file", type=str, required=True, help="H5嵌入文件路径")
    parser.add_argument("--metadata_file", type=str, help="元数据JSON文件路径")
    parser.add_argument("--output_dir", type=str, default="h5_analysis_output", help="输出目录")
    parser.add_argument("--action", type=str, choices=['stats', 'quality', 'search', 'export', 'visualize', 'validate', 'all'], 
                        default='all', help="执行的操作")
    parser.add_argument("--query_id", type=str, help="搜索相似论文的查询ID")
    parser.add_argument("--top_k", type=int, default=10, help="返回最相似的K篇论文")
    parser.add_argument("--use_title", action='store_true', help="使用标题嵌入进行搜索（默认使用摘要）")
    parser.add_argument("--sample_size", type=int, default=1000, help="分析和可视化的采样大小")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger()
    
    # 检查文件是否存在
    if not os.path.exists(args.h5_file):
        logger.error(f"H5文件不存在: {args.h5_file}")
        return
    
    # 创建分析器
    analyzer = H5EmbeddingAnalyzer(args.h5_file, args.metadata_file)
    
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 执行不同的操作
        if args.action in ['stats', 'all']:
            logger.info("获取基本统计信息...")
            stats = analyzer.get_basic_stats()
            print_stats_table(stats, "H5文件基本信息")
            
            # 保存统计信息
            with open(os.path.join(args.output_dir, 'basic_stats.json'), 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
        
        if args.action in ['quality', 'all']:
            logger.info("分析嵌入向量质量...")
            quality_stats = analyzer.analyze_embedding_quality(args.sample_size)
            print_stats_table(quality_stats, "嵌入向量质量分析")
            
            # 保存质量分析结果
            with open(os.path.join(args.output_dir, 'quality_analysis.json'), 'w', encoding='utf-8') as f:
                json.dump(quality_stats, f, ensure_ascii=False, indent=2)
        
        if args.action in ['validate', 'all']:
            logger.info("验证文件完整性...")
            issues = analyzer.validate_integrity()
            if issues:
                print(f"\n❌ 发现 {len(issues)} 个完整性问题:")
                for i, issue in enumerate(issues, 1):
                    print(f"  {i}. {issue}")
            else:
                print("\n✅ 文件完整性验证通过")
            
            # 保存验证结果
            with open(os.path.join(args.output_dir, 'integrity_check.json'), 'w', encoding='utf-8') as f:
                json.dump({'issues': issues, 'is_valid': len(issues) == 0}, f, ensure_ascii=False, indent=2)
        
        if args.action in ['search'] or (args.action == 'search' and args.query_id):
            if not args.query_id:
                logger.error("搜索模式需要提供 --query_id 参数")
            else:
                logger.info(f"搜索与论文 '{args.query_id}' 相似的论文...")
                results, error = analyzer.find_similar_papers(
                    args.query_id, args.top_k, args.use_title
                )
                
                if error:
                    print(f"\n❌ 搜索失败: {error}")
                else:
                    search_type = "标题" if args.use_title else "摘要"
                    print(f"\n🔍 基于{search_type}嵌入的相似论文 (Top {args.top_k}):")
                    print("-" * 60)
                    for result in results:
                        print(f"{result['排名']:2d}. {result['论文ID']} (相似度: {result['相似度']})")
                    
                    # 保存搜索结果
                    with open(os.path.join(args.output_dir, f'search_results_{args.query_id}.json'), 'w', encoding='utf-8') as f:
                        json.dump({'query_id': args.query_id, 'search_type': search_type, 'results': results}, 
                                f, ensure_ascii=False, indent=2)
        
        if args.action in ['export', 'all']:
            logger.info("导出采样数据...")
            export_dir = analyzer.export_sample_data(
                os.path.join(args.output_dir, 'sample_data'), 
                args.sample_size
            )
            print(f"\n📤 采样数据已导出到: {export_dir}")
        
        if args.action in ['visualize', 'all']:
            logger.info("生成可视化图表...")
            analyzer.visualize_embeddings(
                os.path.join(args.output_dir, 'visualizations'), 
                args.sample_size
            )
            print(f"\n📊 可视化图表已保存到: {os.path.join(args.output_dir, 'visualizations')}")
        
        print(f"\n✅ 分析完成! 所有结果已保存到: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"分析过程中出错: {str(e)}")
        raise
    finally:
        analyzer.close()


if __name__ == "__main__":
    main() 