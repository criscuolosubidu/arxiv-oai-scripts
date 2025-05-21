# arXiv-oai-scripts

本项目提供了一套工具，用于处理和分析arXiv论文数据集，以及生成论文摘要和标题的语义嵌入向量。

## 文件结构

```
├── README.md                           # 项目说明文档（英文）
├── README_zh.md                        # 项目说明文档（中文）
├── analyze_arxiv_oai.py                # 数据集质量分析与摘要长度分析工具
├── generate_embeddings_arxiv_oai.py    # 语义嵌入向量生成工具
├── verify_embeddings.py                # 嵌入向量验证工具
├── explore_h5_embeddings.py            # 嵌入向量探索工具
├── compare_embeddings.py               # 嵌入向量比较工具
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

项目包含以下主要工具：

1. **analyze_arxiv_oai.py**: 用于分析arXiv数据集的质量和统计特征，包括检查必要字段、过滤无效数据以及分析摘要长度
2. **generate_embeddings_arxiv_oai.py**: 用于为arXiv论文的标题和摘要生成语义嵌入向量
3. **verify_embeddings.py**: 用于验证生成的嵌入向量与元数据的一致性和质量
4. **explore_h5_embeddings.py**: 用于探索和查看H5文件中的嵌入向量内容
5. **compare_embeddings.py**: 用于比较两个不同的嵌入向量文件的相似度

此外，项目还包含以下辅助工具：
- **download_files.py**: 用于从网络下载必要的数据文件
- **parse_files.py**: 解析处理下载的数据文件
- **unzip_files.py**: 解压缩数据文件
- **search_arxiv_papers.py**: 基于嵌入向量进行论文语义搜索

## 环境依赖

```bash
pip install torch numpy tqdm transformers h5py sentence-transformers matplotlib tabulate scikit-learn
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

### 3. 验证嵌入向量 (verify_embeddings.py)

此工具用于验证H5文件中的嵌入向量是否与元数据一一对应，并且检查嵌入向量的质量。

#### 主要功能

- 检查H5文件和元数据中的论文ID是否一致
- 验证H5文件中ID的顺序与元数据中的索引是否匹配
- 通过重新计算随机样本的嵌入向量来验证嵌入质量
- 计算原始嵌入与重新计算的嵌入之间的余弦相似度

#### 命令行参数

```
--h5_file              H5文件路径，包含嵌入向量（必需）
--metadata_file        元数据JSON文件路径（必需）
--model_path           用于生成嵌入向量的模型路径，必须与原始模型相同（必需）
--num_samples          要验证的随机样本数量（默认：10）
--batch_size           验证时的批处理大小（默认：2）
--log_level            日志级别（默认：INFO）
--use_flash_attention  启用Flash Attention加速（如果可用）
```

#### 使用示例

基础验证：
```bash
python verify_embeddings.py \
    --h5_file data/arxiv/embeddings/arxiv_embeddings_20230101_123456.h5 \
    --metadata_file data/arxiv/embeddings/arxiv_metadata_20230101_123456.json \
    --model_path models/e5-mistral-7b-instruct/
```

增加样本数量和调试级别：
```bash
python verify_embeddings.py \
    --h5_file data/arxiv/embeddings/arxiv_embeddings_20230101_123456.h5 \
    --metadata_file data/arxiv/embeddings/arxiv_metadata_20230101_123456.json \
    --model_path models/e5-mistral-7b-instruct/ \
    --num_samples 20 \
    --log_level DEBUG
```

使用Flash Attention加速：
```bash
python verify_embeddings.py \
    --h5_file data/arxiv/embeddings/arxiv_embeddings_20230101_123456.h5 \
    --metadata_file data/arxiv/embeddings/arxiv_metadata_20230101_123456.json \
    --model_path models/e5-mistral-7b-instruct/ \
    --use_flash_attention
```

### 4. 探索嵌入向量 (explore_h5_embeddings.py)

此工具用于直接探索和查看H5文件中的嵌入向量内容，提供统计信息和可视化功能。

#### 主要功能

- 显示H5文件的基本结构和维度信息
- 计算并显示嵌入向量的统计信息（均值、标准差、范围等）
- 查看单篇论文的详细信息和嵌入向量
- 随机抽样功能，显示多篇论文的基本信息
- 使用PCA进行嵌入向量的可视化（可选）

#### 命令行参数

```
--h5_file              H5文件路径，包含嵌入向量（必需）
--metadata_file        元数据JSON文件路径（必需）
--paper_id             要查看的特定论文ID
--index                要查看的特定论文索引
--samples              要显示的随机样本数量（默认：5）
--plot_pca             绘制嵌入向量的PCA可视化
--log_level            日志级别（默认：INFO）
```

#### 使用示例

基本用法 - 显示随机样本：
```bash
python explore_h5_embeddings.py \
    --h5_file data/arxiv/embeddings/arxiv_embeddings_20230101_123456.h5 \
    --metadata_file data/arxiv/embeddings/arxiv_metadata_20230101_123456.json
```

查看特定论文：
```bash
python explore_h5_embeddings.py \
    --h5_file data/arxiv/embeddings/arxiv_embeddings_20230101_123456.h5 \
    --metadata_file data/arxiv/embeddings/arxiv_metadata_20230101_123456.json \
    --paper_id 1901.00001
```

查看特定索引位置的论文：
```bash
python explore_h5_embeddings.py \
    --h5_file data/arxiv/embeddings/arxiv_embeddings_20230101_123456.h5 \
    --metadata_file data/arxiv/embeddings/arxiv_metadata_20230101_123456.json \
    --index 0
```

绘制PCA可视化：
```bash
python explore_h5_embeddings.py \
    --h5_file data/arxiv/embeddings/arxiv_embeddings_20230101_123456.h5 \
    --metadata_file data/arxiv/embeddings/arxiv_metadata_20230101_123456.json \
    --plot_pca
```

### 5. 比较不同嵌入向量 (compare_embeddings.py)

这个工具用于比较两个不同的H5文件中的嵌入向量，分析它们之间的相似度和差异。特别适用于比较不同模型或参数设置生成的嵌入向量。

#### 主要功能

- 加载并比较两个不同的H5文件中的嵌入向量
- 分析标题和摘要嵌入向量的相似度分布
- 绘制相似度直方图和散点图
- 找出相似度最高和最低的文档案例
- 详细诊断嵌入向量的有效性问题

#### 命令行参数

```
--h5_file1             第一个H5文件路径（必需）
--h5_file2             第二个H5文件路径（必需）
--metadata_file1       第一个元数据JSON文件路径（必需）
--metadata_file2       第二个元数据JSON文件路径（必需）
--output_dir           输出目录路径，用于保存图表等结果
--num_samples          要比较的随机样本数量，不指定则比较所有共同ID
--extreme_cases        显示相似度最高和最低的论文数量（默认：5）
--log_level            日志级别（默认：INFO）
```

#### 使用示例

基本比较：
```bash
python compare_embeddings.py \
    --h5_file1 data/arxiv/embeddings/arxiv_embeddings_model1.h5 \
    --h5_file2 data/arxiv/embeddings/arxiv_embeddings_model2.h5 \
    --metadata_file1 data/arxiv/embeddings/arxiv_metadata_model1.json \
    --metadata_file2 data/arxiv/embeddings/arxiv_metadata_model2.json
```

保存比较结果图表：
```bash
python compare_embeddings.py \
    --h5_file1 data/arxiv/embeddings/arxiv_embeddings_model1.h5 \
    --h5_file2 data/arxiv/embeddings/arxiv_embeddings_model2.h5 \
    --metadata_file1 data/arxiv/embeddings/arxiv_metadata_model1.json \
    --metadata_file2 data/arxiv/embeddings/arxiv_metadata_model2.json \
    --output_dir data/arxiv/comparisons/
```

仅比较部分样本：
```bash
python compare_embeddings.py \
    --h5_file1 data/arxiv/embeddings/arxiv_embeddings_model1.h5 \
    --h5_file2 data/arxiv/embeddings/arxiv_embeddings_model2.h5 \
    --metadata_file1 data/arxiv/embeddings/arxiv_metadata_model1.json \
    --metadata_file2 data/arxiv/embeddings/arxiv_metadata_model2.json \
    --num_samples 100
```

显示更多极端案例：
```bash
python compare_embeddings.py \
    --h5_file1 data/arxiv/embeddings/arxiv_embeddings_model1.h5 \
    --h5_file2 data/arxiv/embeddings/arxiv_embeddings_model2.h5 \
    --metadata_file1 data/arxiv/embeddings/arxiv_metadata_model1.json \
    --metadata_file2 data/arxiv/embeddings/arxiv_metadata_model2.json \
    --extreme_cases 10
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

### 比较分析输出

- `similarity_histogram.png`: 相似度分布直方图
- `similarity_scatter.png`: 标题与摘要相似度的散点图

## 工作流程建议

1. 先使用`analyze_arxiv_oai.py`检查数据质量并过滤无效数据
2. 对过滤后的数据进行摘要长度分析，了解数据特征
3. 使用`generate_embeddings_arxiv_oai.py`为高质量数据生成嵌入向量
4. 用`verify_embeddings.py`验证生成的嵌入向量质量
5. 使用`explore_h5_embeddings.py`探索嵌入向量的特性
6. 如需比较不同模型或参数生成的嵌入向量，使用`compare_embeddings.py`
7. 基于验证良好的嵌入向量构建语义搜索或推荐系统

## 性能优化提示

- 摘要长度分析支持多进程处理，推荐在大型数据集上使用
- 嵌入向量生成可通过调整`batch_size`和`num_workers`优化性能
- 对于非常大的数据集，建议使用`--max_samples`先在子集上测试
- HDF5格式更节省存储空间，NumPy格式更灵活但占用空间较大
- 在比较大型嵌入向量文件时，可以使用`--num_samples`选项限制样本数量，提高比较速度

## 常见问题排查

如果在使用`verify_embeddings.py`时看到类似以下的错误：

```
RuntimeWarning: invalid value encountered in scalar divide
  title_cos_sim = np.dot(stored_title_emb, computed_title_emb) / (
```

或者相似度为`nan`，可能是因为：

1. 嵌入向量中存在全零向量或范数接近零的向量
2. 重新计算的嵌入向量与存储的嵌入向量差异较大

此时，可以使用`compare_embeddings.py`工具来进行更详细的分析，它提供了更强大的向量有效性检查和错误处理机制。 