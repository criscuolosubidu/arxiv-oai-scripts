# arXiv OAI Scripts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> 🚀 高效处理arXiv OAI开放数据集的工具集，专注于论文摘要和标题的语义向量生成与检索

## 📖 项目简介

本项目提供了一套完整的工具链，用于处理arXiv OAI开放数据集，主要功能包括：

- 📄 **论文元数据分析和处理** - 支持数据质量检查、长度分布分析
- 🧠 **高质量语义向量生成** - 支持多种推理后端（TEI、sentence-transformer, transformer, vllm）
- 🔍 **向量质量验证和分析** - 提供多种验证策略和详细统计
- 🗄️ **Qdrant向量数据库集成** - 支持高效语义搜索和多进程导入
- ⚡ **性能优化** - GPU加速、并发处理、内存管理

## 📦 数据集下载

从Kaggle下载arXiv数据集：
- [ArXiv Kaggle OAI Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)

## 🤖 推荐模型

### 默认模型：`intfloat/e5-mistral-7b-instruct`

**优势：**
- ✅ 支持4096 tokens的长文本输入
- ✅ 优秀的嵌入质量，适合学术文本
- ✅ 兼容TEI推理引擎优化
- ✅ 支持指令格式的查询增强

**硬件要求：**
- 🔧 显存：≥18GB（使用flash-attention）
- 💡 如果硬件条件允许，强烈推荐使用此模型

### 替代方案
- `jina-embedding-v3`：适合中文文本处理
- 其他sentence-transformers兼容模型

## 📁 项目结构

```
├── 📋 Cargo.toml                        # 项目配置
├── 📄 LICENSE                           # MIT许可证
├── 📖 README.md                         # 项目文档
├── 🔍 analyze_arxiv_oai.py              # 元数据分析工具
├── 📊 analyze_h5_embeddings.py          # 向量文件分析
├── ✅ check_h5_embeddings.py            # H5向量校验
├── ✅ check_qdrant_or_h5_embeddings.py  # 统一向量校验工具
├── 🔄 compare_embeddings_backend.py     # 后端对比测试
├── 📂 data/
│   ├── arxiv/                           # arXiv数据存储
│   └── pubmed/                          # PubMed数据（规划中）
├── ⬇️ download_files_pubmed.py          # PubMed文件下载工具
├── 🔍 explore_h5_embeddings.py          # 向量探索工具
├── 🔧 find_failed_papers.py             # 错误文件查找
├── 🚀 generate_embeddings_arxiv_oai.py  # 向量生成（transformers）
├── ⚡ generate_embeddings_tei.py        # 向量生成（TEI）
├── 📝 logs/                             # 日志文件
├── 🔗 merge_h5_files.py                 # H5文件合并
├── 🤖 models/
│   ├── e5-mistral-7b-instruct/          # 默认英文模型
│   └── jina-embedding-v3/               # 中文模型
├── 📄 parse_files_pubmed.py             # PubMed文件解析
├── 📋 requirements.txt                  # Python依赖
├── 🔍 search_arxiv_papers.py            # 论文搜索
├── 🗄️ import_to_qdrant.py               # 向量导入Qdrant
├── 🔍 search_with_qdrant.py             # Qdrant语义搜索                      
├── 🧪 test_sampling.py                  # 采样测试工具
├── 📦 unzip_files_pubmed.py             # PubMed解压工具
└── ✅ verify_embeddings.py              # 向量验证
```

## 🛠️ 核心脚本说明

| 脚本名称 | 功能描述 | 使用场景 |
|---------|---------|---------|
| `analyze_arxiv_oai.py` | 分析下载的元数据完整性，统计标题和摘要长度分布 | 数据预处理 |
| `generate_embeddings_tei.py` | 使用TEI引擎生成向量（**推荐**） | 高效生成 |
| `generate_embeddings_with_vllm.py` | 使用vllm生成向量（**推荐**） | 高效生成 |
| `generate_embeddings_arxiv_oai.py` | 使用sentence-transformers/transformers生成向量 | 向量生成 |
| `check_h5_embeddings.py` | 校验H5文件中的向量质量 | 质量控制 |
| `check_qdrant_or_h5_embeddings.py` | 统一的向量校验工具，支持多种采样策略 | 质量控制 |
| `compare_embeddings_backend.py` | 对比不同推理后端的嵌入差异和精度 | 性能测试 |
| `analyze_h5_embeddings.py` | 分析生成的向量文件（H5格式） | 质量评估 |
| `find_failed_papers.py` | 后处理工具，查找生成失败的文件 | 错误排查 |
| `merge_h5_files.py` | 合并多个H5向量文件 | 数据整合 |
| `import_to_qdrant.py` | 将H5向量文件导入Qdrant向量数据库（多进程） | 向量存储 |
| `search_with_qdrant.py` | 使用Qdrant进行语义搜索 | 向量检索 |

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/criscuolosubidu/arxiv-oai-scripts.git
cd arxiv-oai-scripts

# 安装Python依赖（推荐使用conda，并且python版本小于等于3.10.16
pip install -r requirements.txt

### 2. 部署TEI推理引擎（推荐）

使用官方Docker部署[text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference)：

> ⚠️ **重要提示**：对于`e5-mistral-7b-instruct`模型，我们使用平均池化（mean pooling）而非配置文件中的last-token池化方法。

```bash
model="./data/e5-mistral-7b-instruct"
volume="$PWD/data"
docker run --gpus all -p 8080:80 -v $volume:/data \
    --name text-embeddings-inference \
    --pull always ghcr.io/huggingface/text-embeddings-inference:89-1.7 \
    --model-id $model \
    --pooling mean
```

### 3. 数据分析（可选但推荐）

在生成向量之前，建议先分析数据集：

```bash
# 分析数据质量和长度分布
python analyze_arxiv_oai.py \
    --input_file data/arxiv-metadata-oai-snapshot.json \
    --check_and_filter \
    --analyze_length \
    --sample_size 10000
```

### 4. 生成语义向量

#### 使用TEI引擎（推荐）

```bash
python generate_embeddings_tei.py \
    --input_file your_input_arxiv_file.json \
    --output_dir data/arxiv/embeddings \
    --batch_size 100 \
    --max_concurrent 32 \
    --memory_limit_mb 16384
```

#### 使用transformers库

```bash
python generate_embeddings_arxiv_oai.py \
    --input_file your_input_arxiv_file.json \
    --output_dir data/arxiv/embeddings \
    --model_path models/e5-mistral-7b-instruct \
    --use_transformers \
    --batch_size 8 \
    --storage_format h5
```

## 📊 性能优化建议

### 硬件配置
- **GPU**：推荐RTX 4090或更高配置（≥18GB显存）
- **CPU**：多核心处理器，用于并发处理
- **内存**：≥32GB，支持大批量处理

### 参数调优
- `batch_size`：根据显存大小调整，越大GPU利用率越高
- `max_concurrent`：设置为CPU核心数，平衡并发和内存使用
- `memory_limit_mb`：防止内存溢出，根据系统内存设置

## 🗄️ Qdrant向量数据库集成

本项目支持将生成的向量导入到Qdrant向量数据库中，实现高效的语义搜索功能。

### 前置条件

#### 1. 启动Qdrant服务

**AMD GPU用户：**

```bash
docker run \
    --rm \
    --device /dev/kfd --device /dev/dri \
    -p 6333:6333 \
    -p 6334:6334 \
    -e QDRANT__LOG_LEVEL=debug \
    -e QDRANT__GPU__INDEXING=1 \
    qdrant/qdrant:gpu-amd-latest
```

**NVIDIA GPU用户：**

```bash
docker run \
    --rm \
    --gpus=all \
    -p 6333:6333 \
    -p 6334:6334 \
    -e QDRANT__GPU__INDEXING=1 \
    qdrant/qdrant:gpu-nvidia-latest
```

#### 2. 验证Qdrant服务

访问 http://localhost:6333/dashboard 查看Qdrant管理界面。

### 导入向量数据

#### 多进程并行导入（推荐）

```bash
python import_to_qdrant.py \
    --h5_file data/arxiv/embeddings/arxiv_embeddings_20241201_123456.h5 \
    --metadata_file data/arxiv/embeddings/arxiv_metadata_20241201_123456.json \
    --batch_size 500 \
    --num_processes 16 \
    --recreate_collection
```

#### 完整参数示例

```bash
python import_to_qdrant.py \
    --h5_file data/arxiv/embeddings/arxiv_embeddings_20241201_123456.h5 \
    --metadata_file data/arxiv/embeddings/arxiv_metadata_20241201_123456.json \
    --qdrant_url http://localhost:6333 \
    --collection_name arxiv_papers \
    --batch_size 500 \
    --start_index 0 \
    --max_points 10000 \
    --recreate_collection \
    --use_title \
    --use_abstract \
    --distance_metric Cosine \
    --num_processes 16 \
    --timeout 300
```

#### 参数说明

- `--h5_file`: H5嵌入向量文件路径（必需）
- `--metadata_file`: 元数据JSON文件路径（可选，但推荐）
- `--qdrant_url`: Qdrant服务URL（默认: http://localhost:6333）
- `--collection_name`: 集合名称（默认: arxiv_papers）
- `--batch_size`: 批量导入大小（默认: 500）
- `--start_index`: 开始导入的索引位置（默认: 0，用于断点续传）
- `--max_points`: 最大导入点数（可选，用于测试）
- `--recreate_collection`: 重新创建集合（删除现有数据）
- `--use_title`: 导入标题向量（默认: True）
- `--use_abstract`: 导入摘要向量（默认: True）
- `--distance_metric`: 距离度量方式（Cosine/Euclidean/Dot，默认: Cosine）
- `--num_processes`: 并行进程数（默认: CPU核心数）
- `--timeout`: Qdrant客户端超时时间（秒）

### 语义搜索

#### 基本搜索

```bash
python search_with_qdrant.py \
    --query "machine learning transformer attention mechanism" \
    --model_path models/e5-mistral-7b-instruct
```

#### 完整搜索示例

```bash
python search_with_qdrant.py \
    --query "deep learning for natural language processing" \
    --qdrant_url http://localhost:6333 \
    --collection_name arxiv_papers \
    --model_path models/e5-mistral-7b-instruct \
    --vector_name title \
    --top_k 10 \
    --score_threshold 0.7
```

#### 搜索参数说明

- `--query`: 搜索查询文本（必需）
- `--qdrant_url`: Qdrant服务URL
- `--collection_name`: 集合名称
- `--model_path`: 嵌入模型路径（必须与生成向量时使用的模型相同）
- `--vector_name`: 使用的向量类型（title或abstract）
- `--top_k`: 返回结果数量
- `--score_threshold`: 相似度阈值

## 🔧 高级功能

### 向量质量验证

项目提供多种向量质量验证策略：

```bash
# 随机采样验证
python check_h5_embeddings.py \
    --h5_file your_embeddings.h5 \
    --original_metadata_file your_metadata.json \
    --sampling_strategy random \
    --num_samples 1000

# 指数衰减采样（偏向后期数据）
python check_h5_embeddings.py \
    --h5_file your_embeddings.h5 \
    --original_metadata_file your_metadata.json \
    --sampling_strategy exponential_decay \
    --decay_strength 0.8 \
    --num_samples 1000

# 线性衰减采样
python check_h5_embeddings.py \
    --h5_file your_embeddings.h5 \
    --original_metadata_file your_metadata.json \
    --sampling_strategy linear_decay \
    --decay_strength 0.6 \
    --num_samples 1000
```

### 多后端性能对比

```bash
# 对比不同推理后端的性能
python compare_embeddings_backend.py \
    --input_file test_data.json \
    --models models/e5-mistral-7b-instruct \
    --tei_url http://localhost:8080/embed \
    --sample_size 100
```

### H5文件合并

```bash
# 合并多个H5向量文件
python merge_h5_files.py \
    --input_files file1.h5 file2.h5 file3.h5 \
    --output_file merged_embeddings.h5 \
    --compression_level 9
```

### 失败文件查找

```bash
# 查找处理失败的论文
python find_failed_papers.py \
    --input_dir data/arxiv/embeddings \
    --original_metadata data/arxiv-metadata-oai-snapshot.json \
    --output_file failed_papers.json
```

## 📈 监控和日志

所有脚本都支持详细的日志记录：

```bash
# 设置日志级别
python generate_embeddings_tei.py \
    --input_file your_file.json \
    --output_dir output \
    --log_level DEBUG

# 日志文件位置
ls logs/
# tei_embedding_generation_v3_20241201_123456.log
# qdrant_import_mp_20241201_123456.log
# check_embeddings_20241201_123456.log
```

## 🐛 故障排除

### 常见问题

#### 1. TEI服务连接问题

```bash
# 检查TEI服务状态
curl http://localhost:8080/health

# 查看Docker容器日志
docker logs text-embeddings-inference
```

#### 2. 内存不足

- 减小`batch_size`参数
- 增加`memory_limit_mb`限制
- 使用`max_concurrent`控制并发数

#### 3. GPU内存不足

- 减小TEI服务的`--max-batch-tokens`
- 降低`batch_size`
- 使用gradient checkpointing

#### 4. Qdrant连接问题

```bash
# 检查Qdrant服务状态
curl http://localhost:6333/health

# 查看集合信息
curl http://localhost:6333/collections
```

### 性能调优

#### GPU利用率优化

```bash
# 监控GPU使用情况
nvidia-smi -l 1

# 调整TEI服务参数
docker run --gpus all -p 8080:80 \
    --name text-embeddings-inference \
    ghcr.io/huggingface/text-embeddings-inference:89-1.7 \
    --model-id your-model \
    --pooling mean \
    --max-batch-tokens 16384 \
    --max-concurrent-requests 512
```

#### 内存使用优化

```bash
# 监控内存使用
htop

# 使用内存映射文件
python generate_embeddings_tei.py \
    --input_file large_file.json \
    --memory_limit_mb 8192 \
    --batch_size 50
```

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 开发环境设置

```bash
# 克隆项目
git clone https://github.com/criscuolosubidu/arxiv-oai-scripts.git
cd arxiv-oai-scripts

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -r requirements.txt
pip install black flake8 pytest

### 提交流程

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request


## 📄 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情。

## 🙏 致谢

- [Hugging Face](https://huggingface.co/) - 提供优秀的模型和推理引擎
- [arXiv](https://arxiv.org/) - 提供开放的学术数据集
- [Qdrant](https://qdrant.tech/) - 提供高性能向量数据库解决方案
- [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) - 高性能推理引擎
- 所有贡献者和使用者

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 📧 [提交Issue](https://github.com/criscuolosubidu/arxiv-oai-scripts/issues)
- 💬 [参与Discussions](https://github.com/criscuolosubidu/arxiv-oai-scripts/discussions)

## 🔗 相关链接

- [arXiv OAI-PMH接口](https://arxiv.org/help/oa/index)
- [Sentence Transformers文档](https://www.sbert.net/)
- [Qdrant文档](https://qdrant.tech/documentation/)
- [TEI文档](https://huggingface.co/docs/text-embeddings-inference/index)

---

⭐ 如果这个项目对您有帮助，请给一个星标！
