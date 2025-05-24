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

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd arxiv-oai-scripts

# 安装依赖
pip install -r requirements.txt
```

### 2. 部署TEI推理引擎（推荐）

使用官方Docker部署[text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference)：

> ⚠️ **重要提示**：对于`e5-mistral-7b-instruct`模型，我们使用平均池化（mean pooling）而非配置文件中的last-token池化方法。

```bash
# 设置模型路径和挂载目录
model="./models/e5-mistral-7b-instruct"  # 确保模型文件在此目录
volume="$PWD/data"

# 启动TEI服务
docker run --gpus all -p 8080:80 -v $volume:/data \
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
- 所有贡献者和使用者

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 📧 提交Issue
- 💬 参与Discussions

---

⭐ 如果这个项目对您有帮助，请给一个星标！