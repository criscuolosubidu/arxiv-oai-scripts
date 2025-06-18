"""
使用vllm部署qwen3-embeddings-8b来生成嵌入，据测试该模型的效果远好于e5系列
"""

import json
import os
import time
import argparse
import numpy as np
import logging
import h5py
import requests
from datetime import datetime
from tqdm import tqdm
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import threading


# 嵌入使用的instruct
config = {
  "prompts": {
    "query_with_passage": "Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery:",
    "query_with_title": "Instruct: Given a web search query, retrieve relevant titles that answer the query\\nQuery:",
    "query_with_abstract": "Instruct: Given a web search query, retrieve relevant abstracts that answer the query\\nQuery:",
    "query_with_arxiv_paper_abstract": "Instruct: Given a web search query, retrieve relevant arxiv paper abstracts that answer the query\\nQuery:",
    "query_with_arxiv_paper_title": "Instruct: Given a web search query, retrieve relevant arxiv paper titles that answer the query\\nQuery:",
    "title": "Title: ",
    "document": "Passage: ",
    "abstract": "Abstract: ",
    "arxiv_paper_abstract": "Arxiv Paper Abstract: ",
    "arxiv_paper_title": "Arxiv Paper Title: "
  },
  "default_prompt_name": None,
  "similarity_fn_name": "cosine"
}


def setup_logger(log_dir, log_level=logging.INFO):
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"vllm_embedding_generation_v3_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(log_level)

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(file_format)
    logger.addHandler(console_handler)

    return logger


def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB


class StreamingArxivDataset:
    """流式arXiv数据集，支持JSONL和JSON格式，逐行读取，不预加载"""

    def __init__(self, file_path, start_idx=0, max_samples=None):
        self.file_path = file_path
        self.start_idx = start_idx
        self.max_samples = max_samples
        self.file_format = self._detect_file_format()

    def _detect_file_format(self):
        """检测文件格式：JSONL 或 JSON"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if not first_line:
                    raise ValueError("文件为空")

                # 尝试解析第一行
                try:
                    json.loads(first_line)
                    # 检查第二行是否也是有效的JSON对象
                    second_line = f.readline().strip()
                    if second_line:
                        try:
                            json.loads(second_line)
                            return 'jsonl'  # 多行，每行都是JSON对象
                        except json.JSONDecodeError:
                            pass

                    # 只有一行或第二行不是JSON，检查是否是JSON数组
                    f.seek(0)
                    content = f.read().strip()
                    if content.startswith('[') and content.endswith(']'):
                        json.loads(content)  # 验证是否为有效JSON
                        return 'json'
                    else:
                        return 'jsonl'  # 默认为JSONL

                except json.JSONDecodeError:
                    # 第一行不是有效JSON，可能是JSON数组的一部分
                    f.seek(0)
                    content = f.read().strip()
                    if content.startswith('[') and content.endswith(']'):
                        json.loads(content)
                        return 'json'
                    else:
                        raise ValueError("无法识别的文件格式")

        except Exception as e:
            raise ValueError(f"检测文件格式失败: {str(e)}")

    def stream_papers(self):
        """流式迭代论文数据，支持JSONL和JSON格式"""
        if self.file_format == 'jsonl':
            yield from self._stream_jsonl()
        elif self.file_format == 'json':
            yield from self._stream_json()
        else:
            raise ValueError(f"不支持的文件格式: {self.file_format}")

    def _stream_jsonl(self):
        """处理JSONL格式文件"""
        count = 0
        processed = 0

        with open(self.file_path, 'r', encoding='utf-8') as f:
            # 跳过前面的行
            for _ in range(self.start_idx):
                next(f, None)
                count += 1

            for line in f:
                try:
                    paper = json.loads(line)
                    count += 1

                    # 检查必要字段并处理
                    processed_paper = self._process_paper(paper)
                    if processed_paper:
                        yield processed_paper
                        processed += 1

                        if self.max_samples and processed >= self.max_samples:
                            break

                except json.JSONDecodeError:
                    continue

    def _stream_json(self):
        """处理JSON格式文件（list[dict]）"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("JSON文件必须包含一个字典列表")

            processed = 0

            # 应用start_idx和max_samples
            start_idx = self.start_idx
            end_idx = len(data)

            if self.max_samples:
                end_idx = min(start_idx + self.max_samples, len(data))

            for i in range(start_idx, end_idx):
                if i >= len(data):
                    break

                paper = data[i]
                processed_paper = self._process_paper(paper)
                if processed_paper:
                    yield processed_paper
                    processed += 1

                    if self.max_samples and processed >= self.max_samples:
                        break

        except Exception as e:
            raise ValueError(f"处理JSON文件失败: {str(e)}")

    def _process_paper(self, paper):
        """处理单篇论文数据，统一格式"""
        # 检查必要字段
        if 'title' in paper and 'abstract' in paper and paper['title'] and paper['abstract']:
            return {
                'id': paper.get('id', ''),
                'title': paper.get('title', '').replace('\n', ' '),
                'abstract': paper.get('abstract', '').replace('\n', ' '),
                'authors': paper.get('authors', ''),
                'categories': paper.get('categories', ''),
                'journal_ref': paper.get('journal-ref', ''),
                'doi': paper.get('doi', ''),
                'update_date': paper.get('update_date', '')
            }
        return None


def call_vllm_service_sync(text, api_url, prompt_name=None, model_name="qwen3-embeddings-8b", max_retries=3, retry_delay=1):
    """同步调用 vLLM OpenAI 兼容 Embedding API 获取文本嵌入向量"""
    if prompt_name:
        text = prompt_name + text

    payload = {
        "input": text,
        "model": model_name,
        "encoding_format": "float"
    }

    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                resp_json = response.json()
                embedding = np.array(resp_json["data"][0]["embedding"], dtype=np.float16)
                return embedding
            elif response.status_code == 429:
                time.sleep(retry_delay * (2 ** attempt))
                continue
            else:
                logging.error(f"vLLM服务错误，状态码: {response.status_code}")
                raise ValueError(f"vLLM服务返回错误，状态码: {response.status_code}")
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
            else:
                raise ValueError(f"调用 vLLM 服务失败: {str(e)}")

    raise ValueError(f"达到最大重试次数 ({max_retries})")


class HDF5Writer:
    """线程安全的HDF5文件写入器，支持增量写入和压缩"""

    def __init__(self, output_file, embedding_dim):
        self.output_file = output_file
        self.embedding_dim = embedding_dim
        self.h5_file = h5py.File(output_file, 'w')
        self.lock = threading.Lock()

        # 创建可扩展的数据集
        self.title_dataset = self.h5_file.create_dataset(
            'title_embeddings',
            shape=(0, embedding_dim),
            maxshape=(None, embedding_dim),
            dtype=np.float16,
            compression='gzip',
            compression_opts=9,
            chunks=(1000, embedding_dim)
        )

        self.abstract_dataset = self.h5_file.create_dataset(
            'abstract_embeddings',
            shape=(0, embedding_dim),
            maxshape=(None, embedding_dim),
            dtype=np.float16,
            compression='gzip',
            compression_opts=9,
            chunks=(1000, embedding_dim)
        )

        # ID数据集
        id_dtype = h5py.special_dtype(vlen=str)
        self.ids_dataset = self.h5_file.create_dataset(
            'paper_ids',
            shape=(0,),
            maxshape=(None,),
            dtype=id_dtype,
            chunks=(1000,)
        )

        self.current_size = 0

    def append_batch(self, ids, title_embeddings, abstract_embeddings):
        """线程安全地追加一批数据到HDF5文件"""
        with self.lock:
            batch_size = len(ids)
            if batch_size == 0:
                return

            # 扩展数据集
            new_size = self.current_size + batch_size
            self.title_dataset.resize(new_size, axis=0)
            self.abstract_dataset.resize(new_size, axis=0)
            self.ids_dataset.resize(new_size, axis=0)

            # 写入数据
            title_array = np.array(title_embeddings, dtype=np.float16)
            abstract_array = np.array(abstract_embeddings, dtype=np.float16)

            self.title_dataset[self.current_size:new_size] = title_array
            self.abstract_dataset[self.current_size:new_size] = abstract_array
            self.ids_dataset[self.current_size:new_size] = np.array(ids, dtype=object)

            # 立即刷新到磁盘
            self.h5_file.flush()
            self.current_size = new_size

            # 删除临时数组，释放内存
            del title_array, abstract_array

    def close(self):
        """关闭HDF5文件"""
        with self.lock:
            self.h5_file.close()


def process_single_paper_thread(paper, api_url, model_name):
    """线程池中处理单篇论文"""
    try:
        title_embedding = call_vllm_service_sync(paper['title'], api_url, config['prompts']['title'], model_name)
        abstract_embedding = call_vllm_service_sync(paper['abstract'], api_url, config['prompts']['abstract'], model_name)
        return paper['id'], title_embedding, abstract_embedding
    except Exception as e:
        logging.error(f"处理论文 {paper.get('id', '')} 的嵌入时出错: {str(e)}")
        return None


def threaded_process_and_save(
        dataset,
        api_url,
        output_dir,
        batch_size=80,
        prompt_name=None,
        model_name="text-embedding-ada-002",
        max_workers=20,
        memory_limit_mb=2048):
    """使用线程池对论文进行流式处理并保存结果"""

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    embedding_file = os.path.join(output_dir, f"arxiv_embeddings_{timestamp}.h5")
    metadata_file = os.path.join(output_dir, f"arxiv_metadata_{timestamp}.json")

    logging.info(f"最大线程数: {max_workers}")
    logging.info(f"写入批次大小: {batch_size}")
    logging.info(f"内存限制: {memory_limit_mb} MB")

    # 预热以确定嵌入维度
    test_embedding = call_vllm_service_sync("TEST TEXT", api_url, prompt_name, model_name)
    embedding_dim = len(test_embedding)
    logging.info(f"嵌入维度: {embedding_dim}")

    hdf5_writer = HDF5Writer(embedding_file, embedding_dim)

    start_time = time.time()
    total_processed = 0
    failed_count = 0
    completed_results = []

    last_time = start_time
    last_processed = 0

    pbar = tqdm(desc="处理论文", unit="篇")

    pending_futures = set()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for paper in dataset.stream_papers():
                future = executor.submit(process_single_paper_thread, paper, api_url, model_name)
                pending_futures.add(future)

                # 到达线程上限时，先处理已完成的任务
                if len(pending_futures) >= max_workers:
                    done, pending_futures = wait(pending_futures, return_when=FIRST_COMPLETED)

                    for fut in done:
                        result = fut.result()
                        if result:
                            completed_results.append(result)
                            total_processed += 1
                            pbar.update(1)
                        else:
                            failed_count += 1

                # 内存或批次阈值判断
                current_memory = get_memory_usage()
                should_write = (
                        len(completed_results) >= batch_size or
                        current_memory > memory_limit_mb
                )

                if should_write and completed_results:
                    batch_ids = [r[0] for r in completed_results]
                    batch_title_embeddings = [r[1] for r in completed_results]
                    batch_abstract_embeddings = [r[2] for r in completed_results]

                    hdf5_writer.append_batch(batch_ids, batch_title_embeddings, batch_abstract_embeddings)

                    del batch_ids, batch_title_embeddings, batch_abstract_embeddings
                    completed_results.clear()
                    gc.collect()

                    current_time = time.time()
                    if current_time - last_time >= 10:  # 每10秒记录一次吞吐量
                        recent_throughput = (total_processed - last_processed) / (current_time - last_time)
                        memory_mb = get_memory_usage()
                        logging.info(
                            f"已处理 {total_processed} 篇论文 | 失败 {failed_count} 篇 | 当前吞吐量: {recent_throughput:.1f} 篇/秒 | "
                            f"内存使用: {memory_mb:.2f} MB | 活跃线程: {len(pending_futures)}")
                        last_time = current_time
                        last_processed = total_processed

            # 处理剩余未完成任务
            for fut in as_completed(pending_futures):
                result = fut.result()
                if result:
                    completed_results.append(result)
                    total_processed += 1
                    pbar.update(1)
                else:
                    failed_count += 1

            # 写入最后一批
            if completed_results:
                batch_ids = [r[0] for r in completed_results]
                batch_title_embeddings = [r[1] for r in completed_results]
                batch_abstract_embeddings = [r[2] for r in completed_results]

                hdf5_writer.append_batch(batch_ids, batch_title_embeddings, batch_abstract_embeddings)

                completed_results.clear()
                gc.collect()

    except Exception as e:
        logging.error(f"处理过程中出错: {str(e)}")
        hdf5_writer.close()
        raise

    pbar.close()
    hdf5_writer.close()

    final_metadata = {
        'embedding_info': {
            'model': f"vLLM ({model_name})",
            'embedding_dim': embedding_dim,
            'creation_date': datetime.now().isoformat(),
            'prompt_name': prompt_name,
            'total_papers': total_processed,
            'failed_papers': failed_count,
            'success_rate': f"{(total_processed / (total_processed + failed_count) * 100):.2f}%" if (
                    total_processed + failed_count) > 0 else "0%",
            'processing_method': 'thread_pool',
            'data_type': 'float16',
            'compression': 'gzip_level_9',
            'batch_size': batch_size,
            'max_workers': max_workers,
            'memory_limit_mb': memory_limit_mb
        }
    }

    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(final_metadata, f, ensure_ascii=False, indent=2)

    return total_processed, metadata_file, embedding_file


def main():
    parser = argparse.ArgumentParser(description="流式异步arXiv嵌入向量生成")
    parser.add_argument("--input_file", type=str, default="data/arxiv/arxiv-metadata-oai-snapshot.json",
                        help="arXiv元数据文件路径，支持JSONL格式（每行一个JSON对象）和JSON格式（包含字典列表的JSON文件）")
    parser.add_argument("--output_dir", type=str, default="data/arxiv/embeddings",
                        help="嵌入向量输出目录")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="日志输出目录")
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:8000/v1/embeddings",
                        help="vLLM OpenAI兼容Embedding API URL")
    parser.add_argument("--model_name", type=str, default="qwen3-embeddings-8b",
                        help="vLLM部署的模型名称")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="写入批次大小（内存控制，推荐20-100）")
    parser.add_argument("--max_concurrent", type=int, default=20,
                        help="最大并发请求数（并发控制）")
    parser.add_argument("--memory_limit_mb", type=int, default=2048,
                        help="内存使用限制（MB），超过时强制写入")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="从哪篇论文开始处理")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最多处理多少篇论文，None表示处理所有")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    parser.add_argument("--prompt_name", type=str, default=None,
                        help="vLLM服务的提示名称")

    args = parser.parse_args()

    # 设置日志
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(args.log_dir, log_level)

    # 输出运行配置
    logging.info("=" * 60)
    logging.info("使用vLLM(OpenAI兼容)进行arXiv论文标题和摘要的embedding生成")
    logging.info("=" * 60)
    for arg, value in sorted(vars(args).items()):
        logging.info(f"参数 {arg}: {value}")
    logging.info("=" * 60)

    logging.info(f"- 流式处理：实时进度更新")
    logging.info(f"- 写入批次大小: {args.batch_size} 篇")
    logging.info(f"- 最大并发请求: {args.max_concurrent}")
    logging.info(f"- 内存限制: {args.memory_limit_mb} MB")
    logging.info(f"- 预期峰值吞吐量: 接近 50 篇/秒")

    # 测试vLLM服务连接
    test_text = "This is a test text for vLLM service connection."
    try:
        test_embedding = call_vllm_service_sync(test_text, args.api_url, args.prompt_name, args.model_name)
        embedding_dim = test_embedding.shape[0]
        actual_size_per_paper = embedding_dim * 2 * 2  # 2个向量 * 2字节(fp16)
        logging.info(f"vLLM服务连接成功，嵌入维度: {embedding_dim}")
        logging.info(f"实际每篇论文嵌入向量大小: {actual_size_per_paper / 1024:.1f}KB")

        # 重新计算内存使用
        actual_batch_memory = args.batch_size * actual_size_per_paper / 1024 / 1024
        logging.info(f"实际单批次内存占用: {actual_batch_memory:.1f} MB")

    except Exception as e:
        logging.error(f"vLLM服务连接测试失败: {str(e)}")
        return

    # 创建流式数据集
    dataset = StreamingArxivDataset(args.input_file, start_idx=args.start_idx, max_samples=args.max_samples)
    logging.info(f"检测到文件格式: {dataset.file_format.upper()}")

    # 输出初始内存使用情况
    initial_memory = get_memory_usage()
    logging.info(f"初始内存使用: {initial_memory:.2f} MB")

    # 异步处理并保存
    start_time = time.time()
    processed_count, metadata_file, embedding_file = threaded_process_and_save(
        dataset,
        args.api_url,
        args.output_dir,
        batch_size=args.batch_size,
        prompt_name=args.prompt_name,
        model_name=args.model_name,
        max_workers=args.max_concurrent,
        memory_limit_mb=args.memory_limit_mb
    )

    # 输出最终统计
    total_time = time.time() - start_time
    final_memory = get_memory_usage()
    average_throughput = processed_count / total_time

    # 检查输出文件大小
    if os.path.exists(embedding_file):
        file_size_gb = os.path.getsize(embedding_file) / (1024 ** 3)
        logging.info(f"HDF5文件大小: {file_size_gb:.2f} GB")
        compression_ratio = file_size_gb / (processed_count * actual_size_per_paper / (1024 ** 3))
        logging.info(f"压缩率: {compression_ratio:.2f}")

    logging.info("=" * 60)
    logging.info(f"处理完成! 总共处理了 {processed_count} 篇论文")
    logging.info(f"总耗时: {total_time / 60:.2f} 分钟")
    logging.info(f"平均吞吐量: {average_throughput:.2f} 篇/秒")
    logging.info(f"GPU利用率: {average_throughput / 40 * 100:.1f}% (目标:接近100%)")
    logging.info(f"最终内存使用: {final_memory:.2f} MB")
    logging.info(f"内存增长: {final_memory - initial_memory:.2f} MB")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()