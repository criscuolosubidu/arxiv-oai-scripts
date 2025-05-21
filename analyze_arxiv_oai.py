#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析arXiv数据集中摘要的长度分布
计算数据集中最长摘要包含的token数量
数据集来源：https://www.kaggle.com/datasets/Cornell-University/arxiv/data
"""

import json
import os
import argparse
import logging
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import multiprocessing
from functools import partial
import math

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_and_filter_dataset(input_file, output_file=None, required_fields=None, sample_size=None):
    """
    检查arXiv数据集中的字段完整性并过滤不合格的条目
    
    参数:
    - input_file: 输入的JSON文件路径
    - output_file: 过滤后的数据保存路径，如果为None则不保存
    - required_fields: 必须非空的字段列表，默认为['id', 'title', 'abstract']
    - sample_size: 采样大小，为None则处理全部数据
    
    返回:
    - 包含统计信息的字典
    """
    if required_fields is None:
        required_fields = ['id', 'title', 'abstract']
    
    # 统计变量
    total_papers = 0
    valid_papers = 0
    field_missing_stats = {field: 0 for field in required_fields}
    field_empty_stats = {field: 0 for field in required_fields}
    other_fields_stats = {
        'authors': {'missing': 0, 'empty': 0},
        'categories': {'missing': 0, 'empty': 0},
        'journal-ref': {'missing': 0, 'empty': 0},
        'doi': {'missing': 0, 'empty': 0},
        'update_date': {'missing': 0, 'empty': 0}
    }
    
    # 有效的论文数据
    valid_data = []
    
    logging.info(f"开始检查数据集: {input_file}")
    logging.info(f"必须字段: {', '.join(required_fields)}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        # 如果指定了样本大小，先加载全部然后采样
        if sample_size:
            all_lines = f.readlines()
            import random
            if len(all_lines) > sample_size:
                all_lines = random.sample(all_lines, sample_size)
            data_lines = all_lines
        else:
            data_lines = f
        
        # 处理数据
        for line in tqdm(data_lines, desc="检查数据质量"):
            try:
                paper = json.loads(line)
                total_papers += 1
                
                # 检查必须字段
                is_valid = True
                for field in required_fields:
                    if field not in paper:
                        field_missing_stats[field] += 1
                        is_valid = False
                    elif not paper[field]:
                        field_empty_stats[field] += 1
                        is_valid = False
                
                # 检查其他可选字段
                for field in other_fields_stats:
                    if field not in paper:
                        other_fields_stats[field]['missing'] += 1
                    elif not paper[field]:
                        other_fields_stats[field]['empty'] += 1
                
                # 如果所有必须字段都有效，则保留该论文
                if is_valid:
                    valid_papers += 1
                    if output_file:
                        valid_data.append(paper)
                
            except json.JSONDecodeError:
                logging.warning(f"解析JSON失败，跳过一行")
                continue
    
    # 计算总有效率
    valid_rate = (valid_papers / total_papers) * 100 if total_papers > 0 else 0
    
    # 输出统计信息
    logging.info(f"数据集检查完成:")
    logging.info(f"总论文数: {total_papers}")
    logging.info(f"有效论文数: {valid_papers} ({valid_rate:.2f}%)")
    
    logging.info(f"必须字段缺失统计:")
    for field in required_fields:
        missing_rate = (field_missing_stats[field] / total_papers) * 100 if total_papers > 0 else 0
        empty_rate = (field_empty_stats[field] / total_papers) * 100 if total_papers > 0 else 0
        logging.info(f"  {field}: 缺失 {field_missing_stats[field]} ({missing_rate:.2f}%), 空值 {field_empty_stats[field]} ({empty_rate:.2f}%)")
    
    logging.info(f"其他字段统计:")
    for field, stats in other_fields_stats.items():
        missing_rate = (stats['missing'] / total_papers) * 100 if total_papers > 0 else 0
        empty_rate = (stats['empty'] / total_papers) * 100 if total_papers > 0 else 0
        logging.info(f"  {field}: 缺失 {stats['missing']} ({missing_rate:.2f}%), 空值 {stats['empty']} ({empty_rate:.2f}%)")
    
    # 保存过滤后的数据
    if output_file and valid_data:
        logging.info(f"正在保存过滤后的数据到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for paper in valid_data:
                f.write(json.dumps(paper, ensure_ascii=False) + '\n')
        logging.info(f"已保存 {len(valid_data)} 篇有效论文")
    
    # 返回统计信息
    return {
        'total_papers': total_papers,
        'valid_papers': valid_papers,
        'valid_rate': valid_rate,
        'required_field_stats': {
            field: {
                'missing': field_missing_stats[field],
                'missing_rate': (field_missing_stats[field] / total_papers) * 100 if total_papers > 0 else 0,
                'empty': field_empty_stats[field],
                'empty_rate': (field_empty_stats[field] / total_papers) * 100 if total_papers > 0 else 0
            } for field in required_fields
        },
        'other_field_stats': {
            field: {
                'missing': stats['missing'],
                'missing_rate': (stats['missing'] / total_papers) * 100 if total_papers > 0 else 0,
                'empty': stats['empty'],
                'empty_rate': (stats['empty'] / total_papers) * 100 if total_papers > 0 else 0
            } for field, stats in other_fields_stats.items()
        }
    }

def process_paper_batch(papers, tokenizer_name):
    """处理一批论文，计算摘要长度"""
    # 在每个进程中加载分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        logging.error(f"加载分词器失败: {e}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    results = []
    
    for paper in papers:
        abstract = paper.get('abstract', '').replace('\n', ' ')
        tokens = tokenizer.encode(abstract)
        length = len(tokens)
        
        results.append({
            'id': paper.get('id', ''),
            'title': paper.get('title', ''),
            'abstract': abstract,
            'length': length
        })
    
    return results

def analyze_abstract_length(input_file, model_path=None, sample_size=None, num_processes=None):
    """
    分析arXiv论文摘要的长度分布并找出最长的摘要
    
    参数:
    - input_file: 输入的JSON文件路径
    - model_path: 用于分词的模型路径
    - sample_size: 采样大小，为None则处理全部数据
    - num_processes: 进程数，为None则使用CPU核心数-1
    """
    if not model_path:
        model_path = "intfloat/e5-mistral-7b-instruct"
    
    # 确定进程数
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)  # 默认使用CPU核心数-1，保留一个核心
    
    logging.info(f"正在加载分词器...")
    
    # 加载主进程的分词器，主要用于获取max_model_length
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        max_model_length = tokenizer.model_max_length
        logging.info(f"使用分词器: {model_path}, 最大序列长度: {max_model_length}")
    except Exception as e:
        logging.error(f"加载分词器失败: {e}")
        logging.info("使用备用分词器: gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        max_model_length = tokenizer.model_max_length
        logging.info(f"使用备用分词器: gpt2, 最大序列长度: {max_model_length}")
    
    # 加载数据
    logging.info(f"正在处理文件: {input_file}")
    papers = []
    skipped = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="加载数据"):
            try:
                paper = json.loads(line)
                if 'abstract' in paper and paper['abstract']:
                    papers.append(paper)
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1
                continue
    
    # 随机采样
    if sample_size and len(papers) > sample_size:
        import random
        logging.info(f"从数据集中随机采样 {sample_size} 篇论文")
        papers = random.sample(papers, sample_size)
    
    total_papers = len(papers)
    logging.info(f"处理 {total_papers} 篇论文，跳过 {skipped} 条记录")
    
    # 多进程处理
    if num_processes > 1 and total_papers >= num_processes:
        logging.info(f"使用 {num_processes} 个进程并行处理")
        
        # 将数据分割成大小相近的批次
        batch_size = math.ceil(total_papers / num_processes)
        paper_batches = [papers[i:i+batch_size] for i in range(0, total_papers, batch_size)]
        
        # 创建进程池并处理
        with multiprocessing.Pool(processes=num_processes) as pool:
            process_func = partial(process_paper_batch, tokenizer_name=model_path)
            batch_results = list(tqdm(pool.imap(process_func, paper_batches), 
                                      total=len(paper_batches), 
                                      desc="多进程处理"))
        
        # 合并结果
        all_results = []
        for batch in batch_results:
            all_results.extend(batch)
    else:
        # 单进程处理
        logging.info(f"使用单进程处理")
        process_func = partial(process_paper_batch, tokenizer_name=model_path)
        all_results = process_func(papers)
    
    # 整理结果
    abstract_lengths = [result['length'] for result in all_results]
    
    # 找出最长的摘要
    max_result = max(all_results, key=lambda x: x['length']) if all_results else {'id': '', 'title': '', 'abstract': '', 'length': 0}
    max_length = max_result['length']
    max_abstract = max_result['abstract']
    max_title = max_result['title']
    max_id = max_result['id']
    
    # 计算统计信息
    abstract_lengths = np.array(abstract_lengths)
    mean_length = np.mean(abstract_lengths)
    median_length = np.median(abstract_lengths)
    std_length = np.std(abstract_lengths)
    percentiles = [50, 90, 95, 99, 99.9]
    percentile_values = np.percentile(abstract_lengths, percentiles)
    
    # 计算长度分布
    bins = [0, 128, 256, 512, 1024, 2048, 4096, 8192, max_length + 1]
    distribution = []
    for i in range(len(bins) - 1):
        count = np.sum((abstract_lengths >= bins[i]) & (abstract_lengths < bins[i+1]))
        percent = (count / total_papers) * 100
        distribution.append((bins[i], bins[i+1] - 1, count, percent))
    
    # 输出结果
    logging.info(f"总共分析了 {total_papers} 篇论文，跳过了 {skipped} 条记录")
    logging.info(f"摘要token长度统计：")
    logging.info(f"  平均长度: {mean_length:.2f} tokens")
    logging.info(f"  中位数长度: {median_length:.0f} tokens")
    logging.info(f"  标准差: {std_length:.2f} tokens")
    
    logging.info(f"摘要token长度百分位数：")
    for p, v in zip(percentiles, percentile_values):
        logging.info(f"  {p}%: {v:.0f} tokens")
    
    logging.info(f"摘要token长度分布：")
    for start, end, count, percent in distribution:
        logging.info(f"  {start}-{end} tokens: {count} 篇论文 ({percent:.2f}%)")
    
    logging.info(f"最长的摘要：")
    logging.info(f"  ID: {max_id}")
    logging.info(f"  标题: {max_title}")
    logging.info(f"  长度: {max_length} tokens")
    logging.info(f"  摘要: {max_abstract[:500]}..." if len(max_abstract) > 500 else f"  摘要: {max_abstract}")
    
    # 分析模型能够处理的论文比例
    processable = np.sum(abstract_lengths <= max_model_length)
    processable_percent = (processable / total_papers) * 100
    logging.info(f"能被模型完整处理的论文比例 (≤ {max_model_length} tokens): {processable} 篇 ({processable_percent:.2f}%)")
    
    return {
        "total_papers": total_papers,
        "max_length": max_length,
        "max_id": max_id,
        "max_title": max_title,
        "max_abstract": max_abstract,
        "mean_length": mean_length,
        "median_length": median_length,
        "percentiles": dict(zip(percentiles, percentile_values)),
        "distribution": distribution,
        "processable_percent": processable_percent
    }

def main():
    parser = argparse.ArgumentParser(description="分析arXiv论文摘要的长度分布")
    parser.add_argument("--input_file", type=str, default="data/arxiv-metadata-oai-snapshot.json", 
                        help="arXiv元数据JSON文件路径")
    parser.add_argument("--test_file", type=str, default="data/arxiv-mini-test-dataset.jsonl", 
                        help="用于测试的小型数据集文件路径")
    parser.add_argument("--use_test_file", action="store_true",
                        help="使用测试数据集而不是主数据集")
    parser.add_argument("--model_path", type=str, default="intfloat/e5-mistral-7b-instruct", 
                        help="用于分词的模型路径")
    parser.add_argument("--sample_size", type=int, default=None, 
                        help="随机采样的论文数量，为None则处理全部数据")
    parser.add_argument("--output_dir", type=str, default="data", 
                        help="结果输出目录")
    parser.add_argument("--check_and_filter", action="store_true",
                        help="检查并过滤数据集中的无效条目")
    parser.add_argument("--filtered_output", type=str, default=None,
                        help="过滤后的数据集保存路径，仅在--check_and_filter时有效")
    parser.add_argument("--required_fields", type=str, nargs="+", default=["id", "title", "abstract"],
                        help="必须非空的字段列表")
    parser.add_argument("--analyze_length", action="store_true",
                        help="是否分析摘要长度分布")
    parser.add_argument("--num_processes", type=int, default=None,
                        help="用于摘要长度分析的进程数，默认为CPU核心数-1")
    
    args = parser.parse_args()
    
    # 确定要使用的输入文件
    input_file = args.test_file if args.use_test_file else args.input_file
    
    # 确保输入文件存在
    if not os.path.exists(input_file):
        logging.error(f"输入文件不存在: {input_file}")
        return
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成输出文件名，包含时间戳
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_type = "test" if args.use_test_file else "full"
    
    # 如果需要检查和过滤数据
    if args.check_and_filter:
        # 确定过滤后的输出文件
        filtered_output = args.filtered_output
        if not filtered_output:
            filtered_output = os.path.join(args.output_dir, f"filtered_{file_type}_{timestamp}.jsonl")
        
        logging.info(f"开始检查和过滤数据集...")
        filter_results = check_and_filter_dataset(
            input_file, 
            filtered_output, 
            required_fields=args.required_fields,
            sample_size=args.sample_size
        )
        
        # 保存检查结果
        filter_report_file = os.path.join(args.output_dir, f"filter_report_{file_type}_{timestamp}.json")
        with open(filter_report_file, 'w', encoding='utf-8') as f:
            json.dump(filter_results, f, ensure_ascii=False, indent=2)
        
        logging.info(f"数据集检查和过滤完成，报告已保存到: {filter_report_file}")
        
        # 如果用户想要分析过滤后的数据
        if os.path.exists(filtered_output) and os.path.getsize(filtered_output) > 0:
            logging.info(f"是否要使用过滤后的数据继续分析下面流程？[y/N]")
            response = input().strip().lower()
            if response == 'y' or response == 'yes':
                input_file = filtered_output
                logging.info(f"将使用过滤后的数据集: {input_file}")
    
    # 如果需要分析摘要长度
    if args.analyze_length:
        logging.info("开始分析文章摘要长度...")
        
        # 分析文章摘要长度
        output_file = os.path.join(args.output_dir, f"abstract_length_analysis_{file_type}_{timestamp}.json")
        results = analyze_abstract_length(
            input_file, 
            args.model_path, 
            args.sample_size,
            args.num_processes
        )
        
        # 保存结果到文件
        logging.info(f"保存分析结果到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            # 将摘要内容格式化为适合JSON的形式
            results_json = results.copy()
            if 'max_abstract' in results_json:
                results_json['max_abstract'] = results_json['max_abstract'][:2000] + "..." if len(results_json.get('max_abstract', '')) > 2000 else results_json.get('max_abstract', '')
            
            # 递归地将NumPy类型转换为Python原生类型
            def convert_numpy_types(obj):
                if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list) or isinstance(obj, tuple):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            # 将所有NumPy类型转换为Python原生类型
            results_json = convert_numpy_types(results_json)
                    
            json.dump(results_json, f, ensure_ascii=False, indent=2)
        
        logging.info(f"分析完成，结果已保存到: {output_file}")
        
        # 将主要结果保存到一个简明的文本文件中
        summary_file = os.path.join(args.output_dir, f"abstract_length_summary_{file_type}_{timestamp}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"arXiv摘要长度分析摘要\n")
            f.write(f"分析时间: {timestamp}\n")
            f.write(f"分析数据集: {'测试集' if args.use_test_file else '完整数据集'}\n")
            f.write(f"分析论文数量: {results['total_papers']}\n\n")
            
            f.write(f"摘要长度统计:\n")
            f.write(f"  最长摘要: {results['max_length']} tokens\n")
            f.write(f"  平均长度: {results['mean_length']:.2f} tokens\n")
            f.write(f"  中位数长度: {results['median_length']:.0f} tokens\n\n")
            
            f.write(f"摘要长度百分位数:\n")
            for p, v in results['percentiles'].items():
                f.write(f"  {p}%: {v:.0f} tokens\n")
            
            f.write(f"\n最长摘要信息:\n")
            f.write(f"  ID: {results['max_id']}\n")
            f.write(f"  标题: {results['max_title']}\n")
            f.write(f"  长度: {results['max_length']} tokens\n")
            
            # 计算处理能力
            max_model_length = 4096  # 默认假设为4096
            if 'processable_percent' in results:
                f.write(f"\n能被模型完整处理的论文比例: {results['processable_percent']:.2f}%\n")
        
        logging.info(f"摘要报告已保存到: {summary_file}")
        
        return results, output_file, summary_file
    else:
        logging.info("跳过摘要长度分析")
        
        if args.check_and_filter:
            return filter_results
        else:
            logging.info("没有执行任何分析操作，请指定 --check_and_filter 或 --analyze_length")
            return None

if __name__ == "__main__":
    # 设置多进程启动方法
    # 在Windows上，spawn是默认方法
    # 在Unix上，需要显式设置spawn方法来避免潜在的问题
    multiprocessing.set_start_method('spawn', force=True)
    main() 