#!/usr/bin/env bash
# SE-KGRec main experiment: YELP 20-core
# Data layout: data/yelp_20-core/{train,val,test,kg_final,relation_list}.txt
#              data/yelp_20-core/llm_embedding/relation_embeddings/...
set -euo pipefail
cd "$(dirname "$0")"

python main.py \
  --dataset yelp_20-core --data_path data/ \
  --gpu_id 0 --seed 2021 --epoch 12 \
  --kg_completion_mode npmi+fusion \
  --use_relation_aware true --use_x_fusion true \
  --fusion_alpha 0.5 --relation_semantic_threshold 0.7 \
  --kgc_filter_topk 5 --kgc_filter_threshold 0.0
