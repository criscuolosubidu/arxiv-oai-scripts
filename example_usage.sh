#!/bin/bash

# arXiv嵌入向量生成和分析工具使用示例
# 作者: Assistant
# 日期: 2024

echo "==================================="
echo "arXiv嵌入向量工具使用示例"
echo "==================================="

# 配置参数
ARXIV_METADATA="data/arxiv/arxiv-metadata-oai-snapshot.json"
OUTPUT_DIR="data/arxiv/embeddings"
LOG_DIR="logs"
TEI_URL="http://127.0.0.1:8080/embed"

echo "步骤 1: 生成嵌入向量"
echo "-----------------------------------"
python generate_embeddings_tei.py \
    --input_file $ARXIV_METADATA \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --tei_url $TEI_URL \
    --batch_size 50 \
    --max_concurrent 20 \
    --memory_limit_mb 2048 \
    --start_idx 0 \
    --max_samples 1000 \
    --log_level INFO

echo -e "\n步骤 2: 分析生成的H5文件"
echo "-----------------------------------"
# 找到最新生成的文件
H5_FILE=$(ls -t $OUTPUT_DIR/arxiv_embeddings_*.h5 | head -1)
METADATA_FILE=$(ls -t $OUTPUT_DIR/arxiv_metadata_*.json | head -1)

if [ -f "$H5_FILE" ]; then
    echo "分析文件: $H5_FILE"
    python process_h5_embeddings.py \
        --h5_file "$H5_FILE" \
        --metadata_file "$METADATA_FILE" \
        --action all \
        --sample_size 1000 \
        --output_dir h5_analysis_results
else
    echo "未找到H5文件，跳过分析步骤"
fi

echo -e "\n步骤 3: 查找失败的论文"
echo "-----------------------------------"
# 找到最新的日志文件
LOG_FILE=$(ls -t $LOG_DIR/tei_embedding_generation_*.log | head -1)

if [ -f "$LOG_FILE" ] && [ -f "$H5_FILE" ]; then
    echo "方法1: 综合分析（日志+H5对比）"
    python find_failed_papers.py \
        --arxiv_metadata $ARXIV_METADATA \
        --h5_file "$H5_FILE" \
        --log_files "$LOG_FILE" \
        --output_dir failed_papers_analysis \
        --start_idx 0 \
        --max_samples 1000 \
        --get_details \
        --generate_retry_script
    
    echo -e "\n如果有失败的论文，可以运行重试脚本:"
    echo "bash failed_papers_analysis/retry_failed_papers.sh"
elif [ -f "$LOG_FILE" ]; then
    echo "方法2: 快速日志分析（不预加载原始数据）"
    python find_failed_papers.py \
        --arxiv_metadata $ARXIV_METADATA \
        --log_files "$LOG_FILE" \
        --output_dir failed_papers_quick_analysis
    
    echo -e "\n如需详细信息，可运行："
    echo "python find_failed_papers.py --arxiv_metadata $ARXIV_METADATA --log_files \"$LOG_FILE\" --get_details --generate_retry_script"
else
    echo "缺少必要文件，跳过失败分析步骤"
fi

echo -e "\n步骤 4: 搜索相似论文示例"
echo "-----------------------------------"
if [ -f "$H5_FILE" ]; then
    # 从H5文件中随机获取一个论文ID进行测试
    echo "从H5文件中随机选择一个论文ID进行相似性搜索..."
    python process_h5_embeddings.py \
        --h5_file "$H5_FILE" \
        --action search \
        --query_id "0704.0001" \
        --top_k 10 \
        --use_title \
        --output_dir similarity_search_results
else
    echo "未找到H5文件，跳过搜索步骤"
fi

echo -e "\n==================================="
echo "使用示例完成!"
echo "==================================="
echo "生成的文件:"
echo "- 嵌入向量: $OUTPUT_DIR/"
echo "- 分析结果: h5_analysis_results/"
echo "- 失败分析: failed_papers_analysis/"
echo "- 搜索结果: similarity_search_results/"
echo "===================================" 