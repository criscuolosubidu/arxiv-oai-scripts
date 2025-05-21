import os
import gzip
import shutil
import glob
from multiprocessing import Pool, cpu_count
import time

def decompress_file(gz_file):
    """解压单个gz文件到目标目录"""
    # 获取输出文件名
    output_file = os.path.join(
        'data/xml', 
        os.path.basename(gz_file).replace('.gz', '')
    )
    
    try:
        with gzip.open(gz_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"成功解压: {gz_file} -> {output_file}")
        return True
    except Exception as e:
        print(f"解压失败 {gz_file}: {str(e)}")
        return False

def main():
    # 确保目标目录存在
    os.makedirs('data/xml', exist_ok=True)
    
    # 获取所有gz文件
    gz_files = glob.glob('data/xmlgz/*.gz')
    
    if not gz_files:
        print("没有找到gz文件！")
        return
    
    print(f"找到{len(gz_files)}个gz文件等待解压...")
    
    # 确定进程数量 (使用所有可用的CPU核心)
    num_processes = cpu_count()
    print(f"使用{num_processes}个进程并行解压...")
    
    # 开始计时
    start_time = time.time()
    
    # 使用进程池并行解压
    with Pool(processes=num_processes) as pool:
        results = pool.map(decompress_file, gz_files)
    
    # 计算结果
    success_count = results.count(True)
    fail_count = results.count(False)
    
    # 打印统计信息
    elapsed_time = time.time() - start_time
    print(f"\n解压完成! 耗时: {elapsed_time:.2f}秒")
    print(f"成功: {success_count}, 失败: {fail_count}")

if __name__ == "__main__":
    main()
