if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=S_Mamba
seq_len=36
e_layers=2
d_model=256
d_state=2
d_ff=256
enc_in=7
dec_in=7
c_out=7
data=custom
root_path="./dataset/ili/"
data_path="national_illness.csv"
features=M
des='Exp'
itr=1

# 不同预测长度对应的学习率（可选）
declare -A lr_map
lr_map[24]=0.00007
lr_map[36]=0.00007
lr_map[48]=0.00005
lr_map[60]=0.00005

# 预测长度列表
declare -a pred_len_list=(24 36 48 60)

# 遍历训练
for pred_len in "${pred_len_list[@]}"; do
    learning_rate=${lr_map[$pred_len]}
    model_id=ili_${seq_len}_${pred_len}
    log_file="./logs/${model_name}_${model_id}.log"
    
    python -u run.py \
      --is_training 1 \
      --root_path "$root_path" \
      --data_path "$data_path" \
      --model_id "$model_id" \
      --model "$model_name" \
      --data "$data" \
      --features "$features" \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --label_len 18 \
      --e_layers $e_layers \
      --enc_in $enc_in \
      --dec_in $dec_in \
      --c_out $c_out \
      --des "$des" \
      --d_model $d_model \
      --d_state $d_state \
      --d_ff $d_ff \
      --itr $itr \
      --learning_rate $learning_rate > "$log_file" 2>&1

    echo "✅ Training completed for ili | pred_len=$pred_len | LR=$learning_rate | Log: $log_file"
done