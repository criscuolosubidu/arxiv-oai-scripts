#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量导入脚本 - 自动分批导入大量论文数据到Qdrant
支持断点续传、进度监控、错误恢复等功能
"""

import os
import sys
import time
import argparse
import subprocess
import logging
from datetime import datetime


def setup_logger(log_dir: str = "logs") -> logging.Logger:
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"batch_import_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def run_import_batch(
    h5_file: str,
    metadata_file: str,
    start_index: int,
    batch_points: int,
    batch_size: int = 500,
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "arxiv_papers",
    recreate_collection: bool = False,
    timeout: int = 300,
    logger: logging.Logger = None
) -> bool:
    """运行单批次导入"""
    
    cmd = [
        "python", "import_to_qdrant.py",  # 修正文件名
        "--h5_file", h5_file,
        "--metadata_file", metadata_file,
        "--start_index", str(start_index),
        "--max_points", str(batch_points),
        "--batch_size", str(batch_size),
        "--qdrant_url", qdrant_url,
        "--collection_name", collection_name,
        "--timeout", str(timeout)
    ]
    
    # 只在需要时添加recreate_collection参数
    if recreate_collection:
        cmd.append("--recreate_collection")
    
    logger.info(f"开始导入批次: 索引 {start_index:,} - {start_index + batch_points - 1:,}")
    logger.info(f"执行命令: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        end_time = time.time()
        
        duration = end_time - start_time
        
        if result.returncode == 0:
            logger.info(f"批次导入成功，耗时: {duration:.2f} 秒")
            logger.info(f"输出: {result.stdout}")
            return True
        else:
            logger.error(f"批次导入失败，返回码: {result.returncode}")
            logger.error(f"错误输出: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"执行导入命令时发生错误: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="批量导入大量论文数据到Qdrant")
    parser.add_argument("--h5_file", type=str, required=True,
                        help="H5嵌入向量文件路径")
    parser.add_argument("--metadata_file", type=str, required=True,
                        help="元数据文件路径")
    parser.add_argument("--total_papers", type=int, required=True,
                        help="总论文数量")
    parser.add_argument("--batch_points", type=int, default=500000,
                        help="每批次导入的论文数量")
    parser.add_argument("--batch_size", type=int, default=500,
                        help="单次上传的批量大小")
    parser.add_argument("--start_batch", type=int, default=0,
                        help="从第几批开始（用于断点续传）")
    parser.add_argument("--qdrant_url", type=str, default="http://localhost:6333",
                        help="Qdrant服务URL")
    parser.add_argument("--collection_name", type=str, default="arxiv_papers",
                        help="Qdrant集合名称")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="日志输出目录")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="遇到错误时继续处理下一批次")
    parser.add_argument("--recreate_collection", action="store_true",
                        help="重新创建集合（仅在第一批次时生效，删除现有数据）")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Qdrant客户端超时时间（秒），默认300秒")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger(args.log_dir)
    
    # 检查文件是否存在
    if not os.path.exists(args.h5_file):
        logger.error(f"H5文件不存在: {args.h5_file}")
        return
    
    if not os.path.exists(args.metadata_file):
        logger.error(f"元数据文件不存在: {args.metadata_file}")
        return
    
    # 计算批次信息
    total_batches = (args.total_papers + args.batch_points - 1) // args.batch_points
    logger.info(f"总论文数: {args.total_papers:,}")
    logger.info(f"每批次论文数: {args.batch_points:,}")
    logger.info(f"总批次数: {total_batches}")
    logger.info(f"从第 {args.start_batch + 1} 批开始")
    
    # 开始批量导入
    successful_batches = 0
    failed_batches = 0
    overall_start_time = time.time()
    
    # 集合处理说明
    if args.recreate_collection:
        logger.warning("⚠️  将在第一批次重新创建集合，这会删除现有数据！")
    else:
        logger.info("✅ 将追加数据到现有集合（如果集合不存在会自动创建）")
    
    for batch_num in range(args.start_batch, total_batches):
        start_index = batch_num * args.batch_points
        
        # 计算当前批次的实际论文数量
        remaining_papers = args.total_papers - start_index
        current_batch_points = min(args.batch_points, remaining_papers)
        
        if current_batch_points <= 0:
            break
        
        logger.info(f"\n{'='*60}")
        logger.info(f"处理第 {batch_num + 1}/{total_batches} 批次")
        logger.info(f"索引范围: {start_index:,} - {start_index + current_batch_points - 1:,}")
        logger.info(f"论文数量: {current_batch_points:,}")
        logger.info(f"{'='*60}")
        
        # 执行当前批次导入
        # 只在第一批次且用户指定时才重新创建集合
        recreate_for_this_batch = args.recreate_collection and batch_num == args.start_batch
        
        if recreate_for_this_batch:
            logger.warning(f"🔄 第一批次将重新创建集合 '{args.collection_name}'")
        elif batch_num == args.start_batch:
            logger.info(f"📝 将追加数据到集合 '{args.collection_name}'")
        
        success = run_import_batch(
            h5_file=args.h5_file,
            metadata_file=args.metadata_file,
            start_index=start_index,
            batch_points=current_batch_points,
            batch_size=args.batch_size,
            qdrant_url=args.qdrant_url,
            collection_name=args.collection_name,
            recreate_collection=recreate_for_this_batch,
            timeout=args.timeout,
            logger=logger
        )
        
        if success:
            successful_batches += 1
            logger.info(f"✅ 第 {batch_num + 1} 批次导入成功")
        else:
            failed_batches += 1
            logger.error(f"❌ 第 {batch_num + 1} 批次导入失败")
            
            if not args.continue_on_error:
                logger.error("遇到错误，停止导入。使用 --continue_on_error 参数可以跳过错误继续处理")
                break
        
        # 显示总体进度
        completed_papers = (batch_num + 1) * args.batch_points
        if completed_papers > args.total_papers:
            completed_papers = args.total_papers
        
        progress = (completed_papers / args.total_papers) * 100
        logger.info(f"总体进度: {completed_papers:,}/{args.total_papers:,} ({progress:.1f}%)")
        
        # 短暂休息，避免过度占用资源
        if batch_num < total_batches - 1:  # 不是最后一批
            logger.info("等待5秒后继续下一批次...")
            time.sleep(5)
    
    # 总结
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    
    logger.info(f"\n{'='*60}")
    logger.info("批量导入完成!")
    logger.info(f"总耗时: {total_duration:.2f} 秒 ({total_duration/3600:.2f} 小时)")
    logger.info(f"成功批次: {successful_batches}")
    logger.info(f"失败批次: {failed_batches}")
    logger.info(f"总批次: {successful_batches + failed_batches}")
    
    if failed_batches > 0:
        logger.warning(f"有 {failed_batches} 个批次导入失败，请检查日志")
    else:
        logger.info("🎉 所有批次导入成功!")


if __name__ == "__main__":
    main() 