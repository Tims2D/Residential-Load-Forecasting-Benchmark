#!/bin/bash

# Stop on first error
set -e

# Set CUDA device for GPU training
export CUDA_VISIBLE_DEVICES=0

# ============================
# Model & Data Configuration
# ============================
model_name=Times2D
task_name=Multivariate_forecasting
root_path_name=./dataset/
data_path_name=household_data_1min.csv
data_name=household_data_1min
random_seed=2024


# ============================
# Experiment Grid
# ============================
seq_len_list=(96 720)
pred_len_list=(1 4 24 48 96 192 338 720)

for seq_len in "${seq_len_list[@]}"; do

    # Log directory: logs/<model>/<data>/<seq_len>
    log_dir="logs/${model_name}/${data_name}/${seq_len}"
    mkdir -p "$log_dir"

    for pred_len in "${pred_len_list[@]}"; do
        echo "Running with seq_len=${seq_len}, pred_len=${pred_len}"

        model_id_name="${model_name}_${data_name}_seq${seq_len}_pred${pred_len}"

        python -u arguments.py \
            --task_name $task_name \
            --random_seed $random_seed \
            --is_training 1 \
            --root_path $root_path_name \
            --data_path $data_path_name \
            --model_id $model_id_name \
            --model $model_name \
            --data $data_name \
            --features S \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 1 \
            --dec_in 1 \
            --c_out 1 \
            --freq m \
            --e_layers 3 \
            --n_heads 8 \
            --d_model 32 \
            --d_ff 32 \
            --dropout 0.5 \
            --fc_dropout 0.25 \
            --patch_len 48 32 16 6 3 \
            --des Exp \
            --lradj TST \
            --train_epochs 50 \
            --patience 3 \
            --top_k 5 \
            --itr 1 \
            --batch_size 32 \
            --target 'Load' \
            --learning_rate 0.0001 \
            > "${log_dir}/${model_id_name}.log"

    done
done