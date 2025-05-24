#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
查找嵌入生成失败的论文ID工具
支持多种方法：
1. 对比原始元数据和H5文件，找出缺失的论文
2. 分析日志文件，提取失败的论文ID
3. 生成重新处理的脚本

使用示例：

1. 仅分析日志文件（最快，不需要预加载）：
   python find_failed_papers.py \
       --arxiv_metadata data/arxiv/arxiv-metadata-oai-snapshot.json \
       --log_files logs/tei_embedding_generation_*.log

2. 对比H5文件和原始数据：
   python find_failed_papers.py \
       --arxiv_metadata data/arxiv/arxiv-metadata-oai-snapshot.json \
       --h5_file data/arxiv/embeddings/arxiv_embeddings_20241201_143022.h5

3. 综合分析（日志+H5对比）并获取详细信息：
   python find_failed_papers.py \
       --arxiv_metadata data/arxiv/arxiv-metadata-oai-snapshot.json \
       --h5_file data/arxiv/embeddings/arxiv_embeddings_20241201_143022.h5 \
       --log_files logs/tei_embedding_generation_*.log \
       --get_details \
       --generate_retry_script

4. 只从日志提取失败ID并生成重试脚本：
   python find_failed_papers.py \
       --arxiv_metadata data/arxiv/arxiv-metadata-oai-snapshot.json \
       --log_files logs/tei_embedding_generation_*.log \
       --generate_retry_script
"""

import json
import h5py
import re
import os
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
import pandas as pd


def setup_logger():
    """设置日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'failed_papers_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class FailedPapersFinder:
    """失败论文查找器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_original_paper_ids(self, arxiv_metadata_file, start_idx=0, max_samples=None):
        """从原始arXiv元数据文件中获取所有论文ID"""
        self.logger.info(f"正在读取原始元数据文件: {arxiv_metadata_file}")
        
        paper_ids = []
        count = 0
        processed = 0
        
        with open(arxiv_metadata_file, 'r', encoding='utf-8') as f:
            # 跳过前面的行
            for _ in range(start_idx):
                next(f, None)
                count += 1
            
            for line in tqdm(f, desc="读取原始数据"):
                try:
                    paper = json.loads(line)
                    count += 1
                    
                    # 检查必要字段（与原始脚本保持一致）
                    if 'title' in paper and 'abstract' in paper and paper['title'] and paper['abstract']:
                        paper_id = paper.get('id', '')
                        if paper_id:
                            paper_ids.append(paper_id)
                            processed += 1
                            
                            if max_samples and processed >= max_samples:
                                break
                                
                except json.JSONDecodeError:
                    continue
        
        self.logger.info(f"原始数据中符合条件的论文数量: {len(paper_ids):,}")
        return paper_ids
    
    def get_h5_paper_ids(self, h5_file_path):
        """从H5文件中获取所有论文ID"""
        self.logger.info(f"正在读取H5文件: {h5_file_path}")
        
        with h5py.File(h5_file_path, 'r') as h5_file:
            paper_ids = h5_file['paper_ids'][:]
            # 解码为字符串
            paper_ids = [pid.decode('utf-8') for pid in paper_ids]
        
        self.logger.info(f"H5文件中的论文数量: {len(paper_ids):,}")
        return paper_ids
    
    def compare_paper_ids(self, original_ids, h5_ids):
        """对比原始数据和H5文件中的论文ID，找出缺失的"""
        self.logger.info("开始对比论文ID...")
        
        original_set = set(original_ids)
        h5_set = set(h5_ids)
        
        # 找出缺失的论文ID
        missing_ids = original_set - h5_set
        
        # 找出额外的论文ID（理论上不应该有）
        extra_ids = h5_set - original_set
        
        stats = {
            '原始数据论文数': len(original_set),
            'H5文件论文数': len(h5_set),
            '缺失论文数': len(missing_ids),
            '额外论文数': len(extra_ids),
            '成功率': f"{(len(h5_set) / len(original_set) * 100):.2f}%" if original_set else "0%"
        }
        
        return {
            'stats': stats,
            'missing_ids': list(missing_ids),
            'extra_ids': list(extra_ids)
        }
    
    def extract_failed_ids_from_logs(self, log_file_paths, error_patterns=None):
        """从日志文件中提取失败的论文ID"""
        if error_patterns is None:
            error_patterns = [
                r'处理论文 ([^\s]+) 时出错',
                r'处理论文 ([^\s]+) 的嵌入时出错',
                r'论文 ([^\s]+) 处理失败',
                r'ERROR.*论文ID: ([^\s]+)',
                r'失败.*论文.*([0-9]{4}\.[0-9]{4,5})',
            ]
        
        failed_ids = set()
        
        for log_file in log_file_paths:
            if not os.path.exists(log_file):
                self.logger.warning(f"日志文件不存在: {log_file}")
                continue
                
            self.logger.info(f"分析日志文件: {log_file}")
            
            with open(log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    for pattern in error_patterns:
                        matches = re.findall(pattern, line)
                        for match in matches:
                            if match.strip():
                                failed_ids.add(match.strip())
        
        self.logger.info(f"从日志中提取到 {len(failed_ids)} 个失败的论文ID")
        return list(failed_ids)
    
    def get_paper_details(self, paper_ids, arxiv_metadata_file):
        """获取指定论文ID的详细信息"""
        self.logger.info(f"获取 {len(paper_ids)} 个论文的详细信息...")
        
        target_ids = set(paper_ids)
        paper_details = []
        
        with open(arxiv_metadata_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="搜索论文详情"):
                try:
                    paper = json.loads(line)
                    paper_id = paper.get('id', '')
                    
                    if paper_id in target_ids:
                        paper_details.append({
                            'id': paper_id,
                            'title': paper.get('title', '').replace('\n', ' '),
                            'abstract': paper.get('abstract', '').replace('\n', ' '),
                            'authors': paper.get('authors', ''),
                            'categories': paper.get('categories', ''),
                            'journal_ref': paper.get('journal-ref', ''),
                            'doi': paper.get('doi', ''),
                            'update_date': paper.get('update_date', '')
                        })
                        
                        target_ids.remove(paper_id)
                        if not target_ids:  # 找到所有目标论文就停止
                            break
                            
                except json.JSONDecodeError:
                    continue
        
        self.logger.info(f"找到 {len(paper_details)} 个论文的详细信息")
        return paper_details
    
    def generate_retry_script(self, failed_paper_details, original_script_args, output_file):
        """生成重新处理失败论文的脚本"""
        self.logger.info(f"生成重试脚本: {output_file}")
        
        # 创建失败论文的临时JSON文件
        temp_json_file = "failed_papers_temp.json"
        
        script_content = f"""#!/bin/bash
# 重新处理失败论文的脚本
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 失败论文数量: {len(failed_paper_details)}

echo "开始重新处理 {len(failed_paper_details)} 个失败的论文..."

# 创建临时的失败论文数据文件
cat > {temp_json_file} << 'EOF'
"""
        
        # 添加失败论文的JSON数据
        for paper in failed_paper_details:
            script_content += json.dumps(paper, ensure_ascii=False) + "\n"
        
        script_content += f"""EOF

# 运行嵌入生成脚本
python generate_embeddings_tei.py \\
    --input_file {temp_json_file} \\
    --output_dir {original_script_args.get('output_dir', 'data/arxiv/embeddings')} \\
    --log_dir {original_script_args.get('log_dir', 'logs')} \\
    --tei_url {original_script_args.get('tei_url', 'http://127.0.0.1:8080/embed')} \\
    --batch_size {original_script_args.get('batch_size', 50)} \\
    --max_concurrent {original_script_args.get('max_concurrent', 20)} \\
    --memory_limit_mb {original_script_args.get('memory_limit_mb', 2048)} \\
    --start_idx 0 \\
    --max_samples {len(failed_paper_details)} \\
    --log_level INFO \\
    {f"--prompt_name {original_script_args['prompt_name']}" if original_script_args.get('prompt_name') else ""}

echo "重新处理完成!"

# 清理临时文件
rm -f {temp_json_file}
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 给脚本添加执行权限
        os.chmod(output_file, 0o755)
        
        self.logger.info(f"重试脚本已生成: {output_file}")
        return output_file


def save_results(results, output_dir):
    """保存分析结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存统计信息
    if 'comparison' in results:
        with open(os.path.join(output_dir, 'comparison_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(results['comparison'], f, ensure_ascii=False, indent=2)
    
    # 保存缺失的论文ID
    if 'missing_ids' in results:
        with open(os.path.join(output_dir, 'missing_paper_ids.txt'), 'w', encoding='utf-8') as f:
            for paper_id in results['missing_ids']:
                f.write(f"{paper_id}\n")
        
        # 保存为JSON格式
        with open(os.path.join(output_dir, 'missing_paper_ids.json'), 'w', encoding='utf-8') as f:
            json.dump(results['missing_ids'], f, ensure_ascii=False, indent=2)
    
    # 保存从日志提取的失败ID
    if 'log_failed_ids' in results:
        with open(os.path.join(output_dir, 'log_failed_ids.txt'), 'w', encoding='utf-8') as f:
            for paper_id in results['log_failed_ids']:
                f.write(f"{paper_id}\n")
        
        with open(os.path.join(output_dir, 'log_failed_ids.json'), 'w', encoding='utf-8') as f:
            json.dump(results['log_failed_ids'], f, ensure_ascii=False, indent=2)
    
    # 保存失败论文的详细信息
    if 'failed_paper_details' in results:
        with open(os.path.join(output_dir, 'failed_papers_details.json'), 'w', encoding='utf-8') as f:
            json.dump(results['failed_paper_details'], f, ensure_ascii=False, indent=2)
        
        # 保存为CSV格式便于查看
        if results['failed_paper_details']:
            df = pd.DataFrame(results['failed_paper_details'])
            df.to_csv(os.path.join(output_dir, 'failed_papers_details.csv'), index=False, encoding='utf-8')


def print_summary(results):
    """打印结果摘要"""
    print("\n" + "="*60)
    print(" 失败论文分析结果摘要")
    print("="*60)
    
    if 'comparison' in results:
        comparison = results['comparison']
        print(f"\n📊 数据对比统计:")
        for key, value in comparison['stats'].items():
            print(f"  {key}: {value}")
    
    if 'missing_ids' in results:
        print(f"\n❌ 缺失论文数量: {len(results['missing_ids'])}")
        if results['missing_ids'] and len(results['missing_ids']) <= 10:
            print("  示例ID:")
            for paper_id in results['missing_ids'][:10]:
                print(f"    - {paper_id}")
        elif len(results['missing_ids']) > 10:
            print("  示例ID (前10个):")
            for paper_id in results['missing_ids'][:10]:
                print(f"    - {paper_id}")
            print(f"    ... 还有 {len(results['missing_ids']) - 10} 个")
    
    if 'log_failed_ids' in results:
        print(f"\n📋 日志中发现的失败论文: {len(results['log_failed_ids'])}")
        if results['log_failed_ids'] and len(results['log_failed_ids']) <= 5:
            print("  失败ID:")
            for paper_id in results['log_failed_ids'][:5]:
                print(f"    - {paper_id}")
    
    if 'retry_script' in results:
        print(f"\n🔄 重试脚本已生成: {results['retry_script']}")
        print("  运行方式: bash " + results['retry_script'])
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="查找嵌入生成失败的论文ID")
    parser.add_argument("--arxiv_metadata", type=str, required=True, help="原始arXiv元数据JSON文件路径")
    parser.add_argument("--h5_file", type=str, help="生成的H5嵌入文件路径")
    parser.add_argument("--log_files", nargs="+", help="日志文件路径（可以指定多个）")
    parser.add_argument("--output_dir", type=str, default="failed_papers_analysis", help="输出目录")
    parser.add_argument("--start_idx", type=int, default=0, help="原始数据的起始索引")
    parser.add_argument("--max_samples", type=int, default=None, help="原始数据的最大样本数")
    parser.add_argument("--generate_retry_script", action='store_true', help="生成重试脚本")
    parser.add_argument("--original_script_args", type=str, help="原始脚本参数的JSON文件路径")
    parser.add_argument("--get_details", action='store_true', help="获取失败论文的详细信息（需要读取原始文件）")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger()
    
    # 创建查找器
    finder = FailedPapersFinder()
    results = {}
    
    try:
        # 判断是否需要预加载原始数据
        need_original_data = (
            args.h5_file and os.path.exists(args.h5_file)  # 需要与H5文件对比
            or args.get_details                             # 需要获取论文详情
            or args.generate_retry_script                   # 需要生成重试脚本
        )
        
        original_ids = None
        if need_original_data:
            logger.info("步骤 1: 获取原始数据中的论文ID...")
            original_ids = finder.get_original_paper_ids(
                args.arxiv_metadata, 
                args.start_idx, 
                args.max_samples
            )
        else:
            logger.info("跳过原始数据预加载（仅分析日志文件）")
        
        # 2. 如果提供了H5文件，进行对比
        if args.h5_file and os.path.exists(args.h5_file):
            if original_ids is None:
                logger.error("需要原始数据来与H5文件对比，但未加载原始数据")
                return
                
            logger.info("步骤 2: 对比H5文件...")
            h5_ids = finder.get_h5_paper_ids(args.h5_file)
            comparison_result = finder.compare_paper_ids(original_ids, h5_ids)
            results['comparison'] = comparison_result
            results['missing_ids'] = comparison_result['missing_ids']
        
        # 3. 如果提供了日志文件，分析日志
        if args.log_files:
            step_num = 2 if not need_original_data else 3
            logger.info(f"步骤 {step_num}: 分析日志文件...")
            log_failed_ids = finder.extract_failed_ids_from_logs(args.log_files)
            results['log_failed_ids'] = log_failed_ids
        
        # 4. 合并所有失败的ID
        all_failed_ids = set()
        if 'missing_ids' in results:
            all_failed_ids.update(results['missing_ids'])
        if 'log_failed_ids' in results:
            all_failed_ids.update(results['log_failed_ids'])
        
        all_failed_ids = list(all_failed_ids)
        
        # 5. 获取失败论文的详细信息（仅在需要时）
        if all_failed_ids and (args.get_details or args.generate_retry_script):
            if original_ids is None:
                logger.warning("无法获取论文详情：未加载原始数据。请添加 --get_details 参数。")
            else:
                step_num = len([x for x in [need_original_data, args.h5_file, args.log_files] if x]) + 1
                logger.info(f"步骤 {step_num}: 获取失败论文的详细信息...")
                failed_details = finder.get_paper_details(all_failed_ids, args.arxiv_metadata)
                results['failed_paper_details'] = failed_details
                
                # 6. 生成重试脚本
                if args.generate_retry_script and failed_details:
                    logger.info(f"步骤 {step_num + 1}: 生成重试脚本...")
                    
                    # 读取原始脚本参数
                    original_script_args = {}
                    if args.original_script_args and os.path.exists(args.original_script_args):
                        with open(args.original_script_args, 'r', encoding='utf-8') as f:
                            original_script_args = json.load(f)
                    
                    retry_script = finder.generate_retry_script(
                        failed_details,
                        original_script_args,
                        os.path.join(args.output_dir, 'retry_failed_papers.sh')
                    )
                    results['retry_script'] = retry_script
        
        # 7. 保存结果
        logger.info("保存分析结果...")
        save_results(results, args.output_dir)
        
        # 8. 打印摘要
        print_summary(results)
        
        logger.info(f"分析完成! 所有结果已保存到: {args.output_dir}")
        
        # 如果只分析了日志，给用户提示
        if not need_original_data and 'log_failed_ids' in results:
            print(f"\n💡 提示：如果需要获取失败论文的详细信息或生成重试脚本，请添加以下参数：")
            print(f"   --get_details           # 获取论文详细信息")
            print(f"   --generate_retry_script # 生成重试脚本")
        
    except Exception as e:
        logger.error(f"分析过程中出错: {str(e)}")
        raise


if __name__ == "__main__":
    main() 