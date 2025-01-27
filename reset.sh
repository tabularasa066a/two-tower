#!/bin/bash
# models 配下の .pth ファイルをすべて削除
find ./models -type f -name "*.pth" -exec rm -f {} +

# models/embeddings 配下の .idx ファイルをすべて削除
find ./models/embeddings -type f -name "*.idx" -exec rm -f {} +

# data/
find ./data/processed/movielens-1m -type f -name "*.csv" -exec rm -f {} +
