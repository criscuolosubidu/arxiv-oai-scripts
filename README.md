# arXiv-oai-scripts

这个项目提供了一套工具，用于处理和分析arXiv论文数据集，以及生成论文摘要和标题的语义嵌入向量。

## 文件结构

```
├── README.md                           # 项目说明文档
├── analyze_arxiv_oai.py                # 数据集质量分析与摘要长度分析工具
├── generate_embeddings_arxiv_oai.py    # 语义嵌入向量生成工具
├── download_files.py                   # 数据集下载工具
├── parse_files.py                      # 数据解析处理工具
├── unzip_files.py                      # 解压缩工具
├── search_arxiv_papers.py              # 论文搜索工具
├── requirements.txt                    # 项目依赖列表
├── data/                               # 数据存储目录
│   ├── arxiv/                          # arXiv数据集
│   └── pubmed/                         # PubMed数据集
├── logs/                               # 日志文件目录
└── models/                             # 预训练模型目录
    ├── e5-mistral-7b-instruct/        # E5-Mistral-7B模型
    └── jina-embedding-v3/             # JINA嵌入模型
```

## 数据来源

arXiv论文数据集可从以下地址获取：
- [Kaggle上的arXiv数据集](https://www.kaggle.com/datasets/Cornell-University/arxiv/data)

## 工具概述

项目包含两个主要工具：

1. **analyze_arxiv_oai.py**: 用于分析arXiv数据集的质量和统计特征，包括检查必要字段、过滤无效数据以及分析摘要长度
2. **generate_embeddings_arxiv_oai.py**: 用于为arXiv论文的标题和摘要生成语义嵌入向量

此外，项目还包含以下辅助工具：
- **download_files.py**: 用于从网络下载必要的数据文件
- **parse_files.py**: 解析处理下载的数据文件
- **unzip_files.py**: 解压缩数据文件
- **search_arxiv_papers.py**: 基于嵌入向量进行论文语义搜索

## 环境依赖

```bash
pip install torch numpy tqdm transformers h5py sentence-transformers
```

或者直接安装requirements.txt中列出的所有依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据集质量分析与过滤 (analyze_arxiv_oai.py)

这个工具可以检查数据集中必要字段的完整性，过滤无效数据，分析摘要长度分布等。

#### 主要功能

- 检查数据集中字段的存在性和非空性
- 过滤不符合要求的数据条目
- 分析论文摘要的token长度分布
- 支持多进程并行处理以提高效率

#### 命令行参数

```
--input_file          arXiv元数据JSON文件路径
--test_file           用于测试的小型数据集文件路径
--use_test_file       使用测试数据集而不是主数据集
--model_path          用于分词的模型路径
--sample_size         随机采样的论文数量
--output_dir          结果输出目录
--check_and_filter    检查并过滤数据集中的无效条目
--filtered_output     过滤后的数据集保存路径
--required_fields     必须非空的字段列表
--analyze_length      是否分析摘要长度分布
--num_processes       用于摘要长度分析的进程数
```

#### 使用示例

基本数据质量检查：
```bash
python analyze_arxiv_oai.py --check_and_filter
```

数据质量检查并自定义必要字段：
```bash
python analyze_arxiv_oai.py --check_and_filter --required_fields id title abstract authors
```

摘要长度分析（使用多进程加速）：
```bash
python analyze_arxiv_oai.py --analyze_length --num_processes 8
```

数据质量检查与摘要长度分析结合使用：
```bash
python analyze_arxiv_oai.py --check_and_filter --analyze_length --num_processes 4
```

对小样本数据进行分析：
```bash
python analyze_arxiv_oai.py --check_and_filter --analyze_length --sample_size 1000
```

使用过滤后的数据集：
```bash
python analyze_arxiv_oai.py --analyze_length --input_file data/filtered_data.jsonl
```

### 2. 生成语义嵌入向量 (generate_embeddings_arxiv_oai.py)

这个工具使用E5-Mistral-7B模型为arXiv论文的标题和摘要生成语义嵌入向量。

#### 主要功能

- 使用高质量语言模型生成论文标题和摘要的嵌入向量
- 支持HDF5和NumPy两种存储格式
- 提供断点续传功能
- 支持自定义批处理大小和处理参数

#### 命令行参数

```
--input_file            arXiv元数据JSON文件路径
--output_dir            嵌入向量输出目录
--log_dir               日志输出目录
--model_path            模型路径
--batch_size            批处理大小
--data_batch_size       数据加载批次大小
--max_seq_length        最大序列长度
--save_every            每处理多少批次保存一次结果
--start_idx             从哪篇论文开始处理
--max_samples           最多处理多少篇论文
--start_batch           从哪个批次开始处理（断点续传）
--log_level             日志级别
--storage_format        嵌入向量存储格式: h5 (HDF5) 或 numpy
--numpy_save_interval   当使用numpy格式时，每处理多少批次保存一次
--num_workers           DataLoader的工作进程数
--use_flash_attention   启用Flash Attention加速
```

#### 使用示例

基本用法：
```bash
python generate_embeddings_arxiv_oai.py --input_file data/arxiv-metadata-oai-snapshot.json --output_dir data/embeddings
```

使用过滤后的高质量数据：
```bash
python generate_embeddings_arxiv_oai.py --input_file data/filtered_data.jsonl
```

自定义模型路径：
```bash
python generate_embeddings_arxiv_oai.py --model_path models/my-model
```

限制处理的样本数量：
```bash
python generate_embeddings_arxiv_oai.py --max_samples 10000
```

断点续传（从某篇论文继续处理）：
```bash
python generate_embeddings_arxiv_oai.py --start_idx 50000
```

使用NumPy格式存储嵌入向量：
```bash
python generate_embeddings_arxiv_oai.py --storage_format numpy
```

调整批处理大小：
```bash
python generate_embeddings_arxiv_oai.py --batch_size 32 --data_batch_size 64
```

## 输出文件

### 数据分析输出

- `filter_report_*.json`: 数据集质量检查报告
- `filtered_*.jsonl`: 过滤后的有效数据
- `abstract_length_analysis_*.json`: 摘要长度分析详细报告
- `abstract_length_summary_*.txt`: 摘要长度分析简要结果

### 嵌入向量输出

- HDF5格式 (`*.h5`): 包含标题和摘要的嵌入向量以及论文ID
- NumPy格式 (`*_part*.npz`): 分批保存的标题和摘要嵌入向量
- 元数据 (`arxiv_metadata_*.json`): 包含论文详细信息的元数据文件

## 工作流程建议

1. 先使用`analyze_arxiv_oai.py`检查数据质量并过滤无效数据
2. 对过滤后的数据进行摘要长度分析，了解数据特征
3. 使用`generate_embeddings_arxiv_oai.py`为高质量数据生成嵌入向量
4. 基于生成的嵌入向量构建语义搜索或推荐系统

## 性能优化提示

- 摘要长度分析支持多进程处理，推荐在大型数据集上使用
- 嵌入向量生成可通过调整`batch_size`和`num_workers`优化性能
- 对于非常大的数据集，建议使用`--max_samples`先在子集上测试
- HDF5格式更节省存储空间，NumPy格式更灵活但占用空间较大
