# arXiv OAI Scripts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> 🚀 高效处理arXiv OAI开放数据集的工具集，专注于论文摘要和标题的语义向量生成

## 📖 项目简介

本项目提供了一套完整的工具链，用于处理arXiv OAI开放数据集，主要功能包括：
- 📄 论文元数据分析和处理
- 🧠 高质量语义向量生成
- 🔍 向量质量验证和分析
- 🚀 支持多种推理后端（TEI、sentence-transformers等）
- 🗄️ Qdrant向量数据库集成，支持高效语义搜索

## 📦 数据集下载

从Kaggle下载arXiv数据集：
- [ArXiv Kaggle OAI Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)

## 🤖 推荐模型

### 默认模型：`intfloat/e5-mistral-7b-instruct`

**优势：**
- ✅ 支持4096 tokens的长文本输入
- ✅ 优秀的嵌入质量
- ✅ 兼容TEI推理引擎优化
- ✅ 适合英文学术文本

**硬件要求：**
- 🔧 显存：≥18GB，使用flash-attention。
- 💡 如果硬件条件允许，强烈推荐使用此模型

### 替代方案
- `jina-embedding-v3`：适合中文文本处理

## 📁 项目结构

```
├── 📋 Cargo.toml
├── 📄 LICENSE
├── 📖 README.md
├── 🔍 analyze_arxiv_oai.py          # 元数据分析工具
├── 📊 analyze_h5_embeddings.py      # 向量文件分析
├── ✅ check_embeddings_tei.py       # TEI向量校验
├── 🔄 compare_embeddings_backend.py # 后端对比测试
├── 📂 data/
│   ├── arxiv/                       # arXiv数据存储
│   └── pubmed/                      # PubMed数据（规划中）
├── ⬇️ download_files.py             # 文件下载工具
├── 📝 example_usage.sh              # 使用示例
├── 🔍 explore_h5_embeddings.py      # 向量探索工具
├── 🔧 find_failed_papers.py         # 错误文件查找
├── 🚀 generate_embeddings_arxiv_oai.py  # 向量生成（transformers）
├── ⚡ generate_embeddings_tei.py    # 向量生成（TEI）
├── 📝 logs/                         # 日志文件
├── 🔗 merge_h5_files.py             # H5文件合并
├── 🤖 models/
│   ├── e5-mistral-7b-instruct/      # 默认英文模型
│   └── jina-embedding-v3/           # 中文模型
├── 📄 parse_files.py                # 文件解析
├── 📋 requirements.txt              # Python依赖
├── 🔍 search_arxiv_papers.py        # 论文搜索
├── 🗄️ import_to_qdrant.py           # 向量导入Qdrant
├── 🔍 search_with_qdrant.py         # Qdrant语义搜索
├── 🚀 run_qdrant.sh                 # Qdrant启动脚本
├── 🦀 src/
│   └── main.rs                      # Rust源码
├── 📦 unzip_files.py                # 解压工具
└── ✅ verify_embeddings.py          # 向量验证
```

## 🛠️ 核心脚本说明

| 脚本名称 | 功能描述 | 使用场景 |
|---------|---------|---------|
| `analyze_arxiv_oai.py` | 分析下载的元数据完整性，统计标题和摘要长度分布 | 数据预处理 |
| `analyze_h5_embeddings.py` | 分析生成的向量文件（H5格式） | 质量评估 |
| `check_embeddings_tei.py` | 校验TEI推理引擎生成的向量 | 质量控制 |
| `compare_embeddings_backend.py` | 对比不同推理后端的嵌入差异和精度 | 性能测试 |
| `generate_embeddings_arxiv_oai.py` | 使用sentence-transformers/transformers生成向量 | 向量生成 |
| `generate_embeddings_tei.py` | 使用TEI引擎生成向量（**推荐**） | 高效生成 |
| `find_failed_papers.py` | 后处理工具，查找生成失败的文件 | 错误排查 |
| `merge_h5_files.py` | 合并多个H5向量文件 | 数据整合 |
| `import_to_qdrant.py` | 将H5向量文件导入Qdrant向量数据库 | 向量存储 |
| `search_with_qdrant.py` | 使用Qdrant进行语义搜索 | 向量检索 |

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/criscuolosubidu/arxiv-oai-scripts.git
cd arxiv-oai-scripts

# 安装依赖
pip install -r requirements.txt
```

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

### 3. 生成语义向量

```bash
python generate_embeddings_tei.py \
    --input_file your_input_arxiv_file \
    --output_dir your_output_directory \
    --batch_size 100000 \      # 更大的batch_size提高GPU利用率，但需要更多内存
    --max_concurrent 32 \      # 建议设置为CPU核心数
    --memory_limit_mb 16384    # 内存限制（MB）
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

#### 1. 安装Qdrant客户端依赖

```bash
pip install qdrant-client
```

或者安装完整的依赖：

```bash
pip install -r requirements.txt
```

#### 2. 启动Qdrant服务

首先拉取Qdrant镜像：

```bash
# 拉取AMD GPU版本镜像
docker pull qdrant/qdrant:gpu-amd-latest

# 或者拉取NVIDIA GPU版本镜像
docker pull qdrant/qdrant:gpu-nvidia-latest
```

**AMD GPU用户（推荐）：**

```bash
# 使用提供的脚本启动（AMD GPU）
sudo ./run_qdrant.sh

# 或者手动启动AMD GPU版本
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

等待Qdrant服务启动完成（通常需要几分钟下载镜像）。

#### 3. 验证Qdrant服务

访问 http://localhost:6333/dashboard 查看Qdrant管理界面，这里可以很方便操作和查看collection的数据。

### 导入向量数据

#### 基本用法

```bash
python import_to_qdrant.py \
    --h5_file data/arxiv/embeddings/arxiv_embeddings_20241201_123456.h5 \
    --metadata_file data/arxiv/embeddings/arxiv_metadata_20241201_123456.json
```

#### 完整参数示例

```bash
python import_to_qdrant.py \
    --h5_file data/arxiv/embeddings/arxiv_embeddings_20241201_123456.h5 \
    --metadata_file data/arxiv/embeddings/arxiv_metadata_20241201_123456.json \
    --qdrant_url http://localhost:6333 \
    --collection_name arxiv_papers \
    --batch_size 100 \
    --start_index 0 \
    --max_points 10000 \
    --recreate_collection \
    --use_title \
    --use_abstract \
    --distance_metric Cosine \
    --log_level INFO
```

#### 参数说明

- `--h5_file`: H5嵌入向量文件路径（必需）
- `--metadata_file`: 元数据JSON文件路径（可选，但推荐）
- `--qdrant_url`: Qdrant服务URL（默认: http://localhost:6333）
- `--collection_name`: 集合名称（默认: arxiv_papers）
- `--batch_size`: 批量导入大小（默认: 100）
- `--start_index`: 开始导入的索引位置（默认: 0，用于断点续传）
- `--max_points`: 最大导入点数（可选，用于测试）
- `--recreate_collection`: 重新创建集合（删除现有数据）
- `--use_title`: 导入标题向量（默认: True）
- `--use_abstract`: 导入摘要向量（默认: True）
- `--distance_metric`: 距离度量方式（Cosine/Euclidean/Dot，默认: Cosine）

#### 断点续传

如果导入过程中断，可以从指定位置继续：

```bash
python import_to_qdrant.py \
    --h5_file your_file.h5 \
    --metadata_file your_metadata.json \
    --start_index 5000  # 从第5000个向量开始
```

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

### Qdrant性能优化建议

#### 1. 导入优化

- **批量大小**: 根据内存情况调整`--batch_size`，通常100-500比较合适
- **GPU加速**: 确保Qdrant启用了GPU索引（`QDRANT__GPU__INDEXING=1`）
- **分批导入**: 对于大型数据集，可以分多次导入

#### 2. 搜索优化

- **向量选择**: 根据查询类型选择合适的向量（title或abstract）
- **阈值调整**: 调整`score_threshold`来平衡结果质量和数量
- **缓存模型**: 避免重复加载嵌入模型

#### 3. 内存管理

- **监控内存**: 导入大量数据时监控系统内存使用
- **分批处理**: 使用`--max_points`参数进行分批测试

### 故障排除

#### 1. Qdrant连接问题

```bash
# 检查Qdrant服务状态
curl http://localhost:6333/health

# 查看Docker容器日志
docker logs <container_id>
```

#### 2. 内存不足

- 减小`--batch_size`参数
- 使用`--max_points`限制导入数量
- 确保有足够的系统内存

#### 3. 向量维度不匹配

确保H5文件中的向量维度与Qdrant集合配置一致。脚本会自动检测向量维度。

#### 4. 模型路径问题

确保搜索时使用的模型路径与生成嵌入向量时使用的模型相同。

### 完整的端到端示例

#### 1. 启动Qdrant

```bash
# AMD GPU用户
sudo ./run_qdrant.sh

# 或者手动启动
docker run \
    --rm \
    --device /dev/kfd --device /dev/dri \
    -p 6333:6333 \
    -p 6334:6334 \
    -e QDRANT__LOG_LEVEL=debug \
    -e QDRANT__GPU__INDEXING=1 \
    qdrant/qdrant:gpu-amd-latest
```

#### 2. 导入向量（测试少量数据）

```bash
python import_to_qdrant.py \
    --h5_file data/arxiv/embeddings/arxiv_embeddings_20241201_123456.h5 \
    --metadata_file data/arxiv/embeddings/arxiv_metadata_20241201_123456.json \
    --max_points 1000 \
    --recreate_collection
```

#### 3. 验证导入

访问 http://localhost:6333/dashboard 查看集合状态

#### 4. 执行搜索

```bash
python search_with_qdrant.py \
    --query "transformer neural networks" \
    --top_k 5
```

#### 5. 生产环境完整导入

```bash
python import_to_qdrant.py \
    --h5_file your_full_dataset.h5 \
    --metadata_file your_metadata.json \
    --batch_size 200 \
    --recreate_collection
```

### 注意事项

1. **数据一致性**: 确保H5文件和元数据文件对应同一批数据
2. **模型一致性**: 搜索时必须使用与生成向量相同的模型
3. **资源监控**: 导入大量数据时监控CPU、内存和磁盘使用情况
4. **备份**: 重要数据建议在导入前进行备份
5. **版本兼容**: 确保qdrant-client版本与Qdrant服务版本兼容

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

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
- 所有贡献者和使用者

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 📧 提交Issue
- 💬 参与Discussions

---

⭐ 如果这个项目对您有帮助，请给一个星标！