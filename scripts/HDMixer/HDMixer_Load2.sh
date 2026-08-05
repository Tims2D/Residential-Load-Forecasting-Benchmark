#!/bin/bash

# Stop on first error (optional but recommended)
set -e

# Set CUDA device for GPU training
export CUDA_VISIBLE_DEVICES=0

# ============================
# Model & Data Configuration
# ============================
model_name=HDMixer
task_name=Multivariate_forecasting
root_path_name=./dataset/
data_path_name=household_data_1min.csv
data_name=household_data_1min
random_seed=2024

label_len=0

# HDMixer-specific
patch_len=16
stride=8

# ============================
# Experiment Grid
# ============================
seq_len_list=(720)
pred_len_list=(720)

for seq_len in "${seq_len_list[@]}"; do

    # Log directory: logs/<model>/<data>/<seq_len>
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
            --features S \
            --seq_len $seq_len \
            --label_len $label_len \
            --pred_len $pred_len \
            --enc_in 1 \
            --dec_in 1 \
            --c_out 1 \
            --freq m \
            --patch_len $patch_len \
            --stride $stride \
            --n_heads 4 \
            --d_model 16 \
            --d_ff 32 \
            --e_layers 1 \
            --head_dropout 0 \
            --dropout 0.3 \
            --train_epochs 50 \
            --patience 3 \
            --batch_size 32 \
            --target 'Load' \
            --learning_rate 0.0001 \
            --itr 1 \
            > "${log_dir}/${model_id_name}.log"

    done
done
