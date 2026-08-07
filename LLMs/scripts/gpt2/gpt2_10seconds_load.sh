#!/bin/bash
set -e

# Fix 1: expose both GPUs (was: CUDA_VISIBLE_DEVICES=0)
export CUDA_VISIBLE_DEVICES=0,1
export TQDM_DISABLE=1

# ============================
# Model & Data Configuration
# ============================
llm_model=GPT2
model_name=gpt2
task_name=Multivariate_forecasting
root_path_name=./dataset/
data_path_name=10seconds_load.csv
data_name=10seconds_load
random_seed=2024

# ============================
# Training params
# ============================
train_epochs=25
patience=2
batch_size=128
learning_rate=0.001

# ============================
# TimeLLM-specific params
# ============================
llama_layers=12
d_model=32
d_ff=64

# ============================
# Data-specific params
# ============================
label_len=0
features=S
enc_in=1
dec_in=1
c_out=1
freq=s
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

        echo "Running TimeLLM with seq_len=${seq_len}, pred_len=${pred_len}"
        model_id_name="${model_name}_${data_name}_seq${seq_len}_pred${pred_len}"

        # Fix 2: replaced python -u [run.py](http://run.py) with accelerate launch run.py
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
            --llm_model $llm_model \
            --learning_rate $learning_rate \
            --train_epochs $train_epochs \
            --patience $patience \
            --llm_layers $llama_layers \
            --itr 1 \
            --des 'Exp' \
            --model_comment 'TimeLLM-10seconds_load' \
            > "${log_dir}/${model_id_name}.log" 2>&1

        kill_gpu_python_processes

    done
done