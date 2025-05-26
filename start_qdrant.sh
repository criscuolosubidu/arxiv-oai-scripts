#!/bin/bash

# 停止并删除现有的qdrant-gpu容器（如果存在）
sudo docker stop qdrant-gpu 2>/dev/null || true
sudo docker rm qdrant-gpu 2>/dev/null || true

# 启动新的Qdrant容器
sudo docker run \
        --name qdrant-gpu \
        --gpus=all \
        -p 6333:6333 \
        -p 6334:6334 \
        -v ./qdrant_storage:/qdrant/storage \
        -e QDRANT__GPU__INDEXING=1 \
        qdrant/qdrant:gpu-nvidia-latest 