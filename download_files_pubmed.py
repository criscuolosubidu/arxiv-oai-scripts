import requests
import os
import logging
import time
import concurrent.futures
import gzip
import shutil
from tqdm import tqdm

BASE_PUBMED_BASELINE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
SAVE_PATH = "./data/xml/"
LOG_PATH = "./logs/"
YEAR = "25" # 2025

# 配置日志
def setup_logging():
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    
    # 创建日志记录器
    logger = logging.getLogger("pubmed_downloader")
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器，设置编码为utf-8
    log_file = os.path.join(LOG_PATH, f"download_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def download_file(url, file_name, output_path, logger):
    """
    下载文件并保存到指定路径
    
    Args:
        url: 文件URL
        file_name: 文件名
        output_path: 输出路径
        logger: 日志记录器
    
    Returns:
        bool: 下载是否成功
    """
    full_path = os.path.join(output_path, file_name)
    
    # 如果文件已存在，跳过下载
    if os.path.exists(full_path):
        logger.info(f"文件已存在，跳过下载: {file_name}")
        return True
    
    try:
        # 下载文件
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 如果请求失败则抛出异常
        
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)
        
        # 将响应内容写入文件
        with open(full_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"成功下载: {file_name}")
        return True
    
    except requests.exceptions.RequestException as e:
        logger.error(f"下载失败 {file_name}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"处理文件时出错 {file_name}: {str(e)}")
        return False

def download_pubmed_files(start_idx=1, end_idx=1274, max_workers=5):
    """
    使用多线程下载PubMed文件
    
    Args:
        start_idx: 起始文件索引
        end_idx: 结束文件索引
        max_workers: 最大线程数
    """
    logger = setup_logging()
    
    # 确保保存目录存在
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    logger.info(f"开始下载PubMed文件 {start_idx} 到 {end_idx}，使用 {max_workers} 个线程")
    
    successful_downloads = 0
    failed_downloads = 0
    
    # 创建线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交下载任务
        future_to_file = {}
        
        for i in range(start_idx, end_idx + 1):
            file_name = f"pubmed{YEAR}n{i:04d}.xml.gz"
            url = f"{BASE_PUBMED_BASELINE_URL}{file_name}"
            
            # 提交任务到线程池
            future = executor.submit(download_file, url, file_name, SAVE_PATH, logger)
            future_to_file[future] = file_name
        
        # 显示进度条
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(future_to_file), desc="下载进度"):
            file_name = future_to_file[future]
            try:
                success = future.result()
                if success:
                    successful_downloads += 1
                else:
                    failed_downloads += 1
            except Exception as e:
                logger.error(f"处理 {file_name} 时发生异常: {str(e)}")
                failed_downloads += 1
    
    logger.info(f"下载完成. 成功: {successful_downloads}, 失败: {failed_downloads}")
    
    # 返回下载结果摘要
    return {
        "successful": successful_downloads,
        "failed": failed_downloads,
        "total": end_idx - start_idx + 1
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='下载PubMed XML文件')
    parser.add_argument('--start', type=int, default=1, help='起始文件索引 (默认: 1)')
    parser.add_argument('--end', type=int, default=1274, help='结束文件索引 (默认: 1274)')
    parser.add_argument('--threads', type=int, default=5, help='下载线程数 (默认: 5)')
    
    args = parser.parse_args()
    
    # 运行下载
    results = download_pubmed_files(args.start, args.end, args.threads)
    
    print(f"\n下载统计:")
    print(f"总文件数: {results['total']}")
    print(f"成功下载: {results['successful']}")
    print(f"下载失败: {results['failed']}")
