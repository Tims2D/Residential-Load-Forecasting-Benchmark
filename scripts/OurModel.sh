#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
# Set parameters
seq_len=720
label_len=48 



model_name=OurModel
task_name=Multivariate_forecasting
root_path_name=./dataset/
data_path_name=Load1.csv
data_name=Load1
model_id_name=Load1
random_seed=2024

# Create necessary directories if they do not exist
log_dir=logs/$model_name

if [ ! -d "$log_dir" ]; then
    mkdir -p $log_dir
fi

pred_len_list=(1 4 24 48 96 192 384 672 1344 2688)
pred_len=384

#for pred_len in "${pred_len_list[@]}"; do
#    echo "Running with pred_len: $pred_len"
model_id_name="${model_name}_${data_name}"

python -u arguments.py \
  --data_path $data_path_name \
  --data $data_name \
  --task_name $task_name \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name \
  --model $model_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 6 \
  --c_out 6 \
  --dec_in 6 \
  --e_layers 3 \
  --n_heads 8 \
  --d_model 16 \
  --d_ff 128 \
  --dropout 0.4 \
  --fc_dropout 0.0 \
  --kernel_list 5 7 11 15 \
  --period 48 90 110 360 720 \
  --patch_len 48 32 16 6 3 \
  --stride 48 32 16 6 3 \
  --trend_kernel_size 25 \
  --seasonal_kernel_size 7\
  --des 'Exp' \
  --train_epochs 50\
  --patience 5 \
  --lradj 'TST' \
  --target 'Load_demand' \
  --top_k 5 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --itr 1 >logs/OurModel/$model_id_name'_'$seq_len'_'$pred_len.log
      

#timeenc 0