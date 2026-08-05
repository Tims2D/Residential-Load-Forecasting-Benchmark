#!/bin/bash

# Stop on first error
set -e

# Set CUDA device for GPU training
export CUDA_VISIBLE_DEVICES=0

# ============================
# Model & Data Configuration
# ============================
model_name=Crossformer
task_name=Multivariate_forecasting
root_path_name=./dataset/
data_path_name=hourly_load.csv
data_name=hourly_load
random_seed=2024

label_len=0

# ============================
# Experiment Grid
# ============================
seq_len_list=(96 720)
pred_len_list=(1 4 24 48 96 192 336 720)

for seq_len in "${seq_len_list[@]}"; do

    log_dir="logs/${model_name}/${data_name}/${seq_len}"
    mkdir -p "$log_dir"

    for pred_len in "${pred_len_list[@]}"; do
        echo "Running with seq_len=${seq_len}, pred_len=${pred_len}"

        model_id_name="${model_name}_${data_name}_seq${seq_len}_pred${pred_len}"

        python -u arguments.py \
            --random_seed $random_seed \
            --task_name $task_name \
            --model_id $model_id_name \
            --is_training 1 \
            --model $model_name \
            --data $data_name \
            --root_path $root_path_name \
            --data_path $data_path_name \
            --features M \
            --seq_len $seq_len \
            --label_len $label_len \
            --pred_len $pred_len \
            --enc_in 12 \
            --dec_in 12 \
            --c_out 12 \
            --freq h \
            --e_layers 2 \
            --n_heads 8 \
            --d_model 32 \
            --d_ff 32 \
            --dropout 0.25 \
            --fc_dropout 0.15 \
            --top_k 5 \
            --train_epochs 50 \
            --patience 3 \
            --batch_size 32 \
            --target 'Load' \
            --learning_rate 0.0001 \
            --itr 1 \
            > "${log_dir}/${model_id_name}.log"

    done
done