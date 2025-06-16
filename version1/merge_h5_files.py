#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
合并多个H5文件的脚本
支持合并包含 title_embeddings, abstract_embeddings, paper_ids 的H5文件
注意，由于加载了所有的数据到内存中，因此内存占用很高，需要90G及以上的内存！
"""

import h5py
import numpy as np
import argparse
import os
import logging
from datetime import datetime
import json


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


def get_h5_info(file_path):
    """获取H5文件的基本信息"""
    try:
        with h5py.File(file_path, 'r') as f:
            info = {}
            for key in f.keys():
                dataset = f[key]
                info[key] = {
                    'shape': dataset.shape,
                    'dtype': str(dataset.dtype),
                    'size': dataset.size
                }
            return info
    except Exception as e:
        logging.error(f"读取文件 {file_path} 信息失败: {str(e)}")
        return None


def check_compatibility(file_paths):
    """检查文件兼容性"""
    logging.info("检查文件兼容性...")
    
    reference_info = None
    file_infos = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        info = get_h5_info(file_path)
        if info is None:
            raise ValueError(f"无法读取文件: {file_path}")
        
        file_infos.append((file_path, info))
        
        # 使用第一个文件作为参考
        if reference_info is None:
            reference_info = info
            logging.info(f"参考文件: {file_path}")
            for key, details in info.items():
                logging.info(f"  {key}: shape={details['shape']}, dtype={details['dtype']}")
        else:
            # 检查数据集名称是否一致
            if set(info.keys()) != set(reference_info.keys()):
                raise ValueError(f"文件 {file_path} 的数据集名称与参考文件不一致")
            
            # 检查维度是否兼容（除了第一维）
            for key in info.keys():
                ref_shape = reference_info[key]['shape']
                curr_shape = info[key]['shape']
                ref_dtype = reference_info[key]['dtype']
                curr_dtype = info[key]['dtype']
                
                if len(ref_shape) != len(curr_shape):
                    raise ValueError(f"文件 {file_path} 中 {key} 的维度数量不匹配")
                
                if len(ref_shape) > 1 and ref_shape[1:] != curr_shape[1:]:
                    raise ValueError(f"文件 {file_path} 中 {key} 的形状不兼容")
                
                if ref_dtype != curr_dtype:
                    logging.warning(f"文件 {file_path} 中 {key} 的数据类型不同: {curr_dtype} vs {ref_dtype}")
    
    return file_infos, reference_info


def merge_h5_files(input_files, output_file, remove_duplicates=True):
    """合并多个H5文件"""
    logging.info(f"开始合并 {len(input_files)} 个H5文件...")
    
    # 检查兼容性
    file_infos, reference_info = check_compatibility(input_files)
    
    # 计算总大小
    total_sizes = {}
    for key in reference_info.keys():
        total_sizes[key] = sum(info[key]['size'] for _, info in file_infos)
    
    logging.info("文件信息:")
    for file_path, info in file_infos:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logging.info(f"  {os.path.basename(file_path)}: {file_size_mb:.2f} MB")
        for key, details in info.items():
            logging.info(f"    {key}: {details['shape'][0]} 条记录")
    
    # 创建输出文件
    logging.info(f"创建输出文件: {output_file}")
    
    with h5py.File(output_file, 'w') as output_h5:
        # 收集所有数据
        all_data = {}
        
        # 初始化数据容器
        for key in reference_info.keys():
            all_data[key] = []
        
        # 如果需要去重，收集所有paper_ids
        seen_ids = set() if remove_duplicates else None
        duplicate_count = 0
        
        # 逐个读取文件并收集数据
        for file_path, info in file_infos:
            logging.info(f"读取文件: {os.path.basename(file_path)}")
            
            with h5py.File(file_path, 'r') as input_h5:
                # 读取paper_ids用于去重检查
                if remove_duplicates and 'paper_ids' in input_h5:
                    paper_ids = input_h5['paper_ids'][:]
                    
                    # 找出非重复的索引
                    valid_indices = []
                    for i, paper_id in enumerate(paper_ids):
                        if isinstance(paper_id, bytes):
                            paper_id = paper_id.decode('utf-8')
                        
                        if paper_id not in seen_ids:
                            seen_ids.add(paper_id)
                            valid_indices.append(i)
                        else:
                            duplicate_count += 1
                    
                    if len(valid_indices) == 0:
                        logging.warning(f"文件 {file_path} 中所有记录都是重复的，跳过")
                        continue
                    
                    logging.info(f"  发现 {len(paper_ids)} 条记录，其中 {len(valid_indices)} 条非重复")
                    
                    # 只读取非重复的数据
                    for key in input_h5.keys():
                        if len(valid_indices) == len(paper_ids):
                            # 没有重复，直接读取全部
                            data = input_h5[key][:]
                        else:
                            # 有重复，只读取非重复的索引
                            data = input_h5[key][valid_indices]
                        all_data[key].append(data)
                else:
                    # 不去重或没有paper_ids，直接读取所有数据
                    record_count = list(input_h5.values())[0].shape[0]
                    logging.info(f"  读取 {record_count} 条记录")
                    
                    for key in input_h5.keys():
                        data = input_h5[key][:]
                        all_data[key].append(data)
        
        if remove_duplicates and duplicate_count > 0:
            logging.info(f"总共去除了 {duplicate_count} 条重复记录")
        
        # 合并数据并写入输出文件
        logging.info("合并数据并写入输出文件...")
        
        final_counts = {}
        for key, data_list in all_data.items():
            if not data_list:
                logging.warning(f"没有数据可合并到 {key}")
                continue
            
            logging.info(f"合并 {key}...")
            
            # 合并数据
            if key == 'paper_ids':
                # 字符串数据特殊处理
                merged_data = np.concatenate(data_list, axis=0)
            else:
                # 数值数据
                merged_data = np.concatenate(data_list, axis=0)
            
            final_counts[key] = len(merged_data)
            
            # 确定数据类型和压缩设置
            if key == 'paper_ids':
                dtype = h5py.special_dtype(vlen=str)
                compression = None
                chunks = (1000,)
            else:
                dtype = merged_data.dtype
                compression = 'gzip'
                chunks = (1000, merged_data.shape[1]) if len(merged_data.shape) > 1 else (1000,)
            
            # 创建数据集并写入
            dataset = output_h5.create_dataset(
                key,
                data=merged_data,
                dtype=dtype,
                compression=compression,
                compression_opts=9 if compression else None,
                chunks=chunks
            )
            
            logging.info(f"  {key}: {merged_data.shape} -> 已写入")
            
            # 释放内存
            del merged_data
        
        # 验证数据一致性
        if len(set(final_counts.values())) > 1:
            logging.error(f"数据集大小不一致: {final_counts}")
            raise ValueError("合并后的数据集大小不一致")
        
        total_records = list(final_counts.values())[0] if final_counts else 0
        logging.info(f"合并完成，总记录数: {total_records}")
    
    return total_records, duplicate_count


def create_merged_metadata(input_files, output_file, total_records, duplicate_count):
    """创建合并后的元数据文件"""
    metadata = {
        'merge_info': {
            'creation_date': datetime.now().isoformat(),
            'input_files': [os.path.basename(f) for f in input_files],
            'total_input_files': len(input_files),
            'total_records': total_records,
            'duplicate_records_removed': duplicate_count,
            'output_file': os.path.basename(output_file)
        },
        'file_details': []
    }
    
    # 添加每个输入文件的详细信息
    for file_path in input_files:
        if os.path.exists(file_path):
            file_info = get_h5_info(file_path)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            detail = {
                'filename': os.path.basename(file_path),
                'size_mb': round(file_size_mb, 2),
                'datasets': file_info
            }
            metadata['file_details'].append(detail)
    
    # 保存元数据
    metadata_file = output_file.replace('.h5', '_metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logging.info(f"元数据已保存到: {metadata_file}")
    return metadata_file


def main():
    parser = argparse.ArgumentParser(description="合并多个H5文件")
    parser.add_argument("input_files", nargs='+', help="要合并的H5文件路径列表")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出文件路径")
    parser.add_argument("--no-remove-duplicates", action="store_true", 
                        help="不去除重复记录（基于paper_ids）")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level)
    setup_logger(log_level)
    
    # 输出配置信息
    logging.info("="*60)
    logging.info("H5文件合并工具")
    logging.info("="*60)
    logging.info(f"输入文件数量: {len(args.input_files)}")
    for i, file_path in enumerate(args.input_files, 1):
        logging.info(f"  {i}. {file_path}")
    logging.info(f"输出文件: {args.output}")
    logging.info(f"去除重复: {'否' if args.no_remove_duplicates else '是'}")
    logging.info("="*60)
    
    try:
        # 检查输出目录
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"创建输出目录: {output_dir}")
        
        # 执行合并
        total_records, duplicate_count = merge_h5_files(
            args.input_files, 
            args.output, 
            remove_duplicates=not args.no_remove_duplicates
        )
        
        # 创建元数据
        metadata_file = create_merged_metadata(
            args.input_files, 
            args.output, 
            total_records, 
            duplicate_count
        )
        
        # 输出最终统计
        output_size_mb = os.path.getsize(args.output) / (1024 * 1024)
        logging.info("="*60)
        logging.info("合并完成!")
        logging.info(f"输出文件: {args.output}")
        logging.info(f"文件大小: {output_size_mb:.2f} MB")
        logging.info(f"总记录数: {total_records}")
        if not args.no_remove_duplicates:
            logging.info(f"去除重复: {duplicate_count} 条")
        logging.info(f"元数据文件: {metadata_file}")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"合并失败: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 