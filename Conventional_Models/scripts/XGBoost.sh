#!/bin/bash

# Set up your environment
export CUDA_VISIBLE_DEVICES=0

# Directories for logs and checkpoints
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/xgb" ]; then
    mkdir ./logs/xgb
fi

# Execute the XGBoost script
python -u XGBoost.py \
    --is_training 1 \
    --random_seed 2024 \
    --data ETTm1 \
    --root_path ./dataset/ \
    --data_path ETTm1.csv \
    --features 'M' \
    --target 'OT' \
    --seq_len 720 \
    --pred_len 96 \
    --n_estimators 100 \
    --max_depth 10 \
    --learning_rate 0.05 \
    --early_stopping_rounds 10 \
    --model_save_path './models/xgb_model.pkl' > logs/xgb/XGBoost_ETTm1.log
