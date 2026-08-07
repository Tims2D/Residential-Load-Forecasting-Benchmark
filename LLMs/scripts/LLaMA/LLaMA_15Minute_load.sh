#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1
export TQDM_DISABLE=1

# ============================
# Model & Data Configuration
# ============================
llm_model=LLaMA
model_name=LLaMA
task_name=Multivariate_forecasting
root_path_name=./dataset/
data_path_name=15Minute_load.csv
data_name=15Minute_load
random_seed=2024

# ============================
# Training params (kept same as gpt2 script)
# ============================
train_epochs=25
patience=2
batch_size=128
learning_rate=0.0001

# ============================
# LLaMA-specific params
# NOTE: llm_dim=4096 — LLaMA-7B hidden dimension
#       d_model and d_ff kept same as gpt2 (reprogramming layer only)
# ============================
llama_layers=6
d_model=32
d_ff=64
llm_dim=4096

# ============================
# Data-specific params
# enc_in/dec_in/c_out = 1 (data_loader returns target only)
# ============================
label_len=0
features=M
enc_in=1
dec_in=1
c_out=1
freq=15min
target='Load'

# ============================
# GPU cleanup helper
# ============================
kill_gpu_python_processes() {
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)
    [ -z "${pids}" ] && return
    for pid in $pids; do
        if ps -p "$pid" -o comm= 2>/dev/null | grep -qi python; then
            echo "  [cleanup] Killing python PID $pid"
            kill "$pid" 2>/dev/null; sleep 1; kill -9 "$pid" 2>/dev/null
        fi
    done
    sleep 5
}

# ============================
# Experiment Grid
# ============================
seq_len_list=(96 720)
pred_len_list=(1 4 24 48 96 192 338 720)

for seq_len in "${seq_len_list[@]}"; do

    log_dir="logs/${model_name}/${data_name}/${seq_len}"
    mkdir -p "$log_dir"

    for pred_len in "${pred_len_list[@]}"; do

        echo "Running LLaMA with seq_len=${seq_len}, pred_len=${pred_len}"
        model_id_name="${model_name}_${data_name}_seq${seq_len}_pred${pred_len}"

        accelerate launch \
            --config_file /u/rnematirad1/LLMs/accelerate_config.yaml \
            run.py \
            --task_name $task_name \
            --is_training 1 \
            --model_id $model_id_name \
            --model $model_name \
            --data $data_name \
            --root_path $root_path_name \
            --data_path $data_path_name \
            --features $features \
            --seq_len $seq_len \
            --label_len $label_len \
            --pred_len $pred_len \
            --enc_in $enc_in \
            --dec_in $dec_in \
            --c_out $c_out \
            --freq $freq \
            --target $target \
            --d_model $d_model \
            --d_ff $d_ff \
            --batch_size $batch_size \
            --learning_rate $learning_rate \
            --train_epochs $train_epochs \
            --patience $patience \
            --llm_model $llm_model \
            --llm_dim $llm_dim \
            --llm_layers $llama_layers \
            --itr 1 \
            --des 'Exp' \
            --model_comment 'LLaMA-15Minute_load' \
            > "${log_dir}/${model_id_name}.log" 2>&1

        kill_gpu_python_processes

    done
done