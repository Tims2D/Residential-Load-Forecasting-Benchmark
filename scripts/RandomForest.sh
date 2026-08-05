#!/bin/bash

# Set up your environment
export CUDA_VISIBLE_DEVICES=0

# Directories for logs and checkpoints
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/rf" ]; then
    mkdir ./logs/rf
fi

# Execute the RandomForest script
python -u RandomForest.py \
    --is_training 1 \
    --random_seed 2024 \
    --data ETTm1 \
    --root_path ./dataset/ \
    --data_path ETTm1.csv \
    --features 'M' \
    --target 'OT' \
    --seq_len 96 \
    --pred_len 96 \
    --n_estimators 100 \
    --max_depth 10 \
    --model_save_path './models/random_forest_model.pkl' > logs/rf/RandomForest_ETTm1.log
