#!/bin/bash

# # 设置 GPU
# export CUDA_VISIBLE_DEVICES=0

# # 创建日志目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# model_name=S_Mamba
# seq_len=96
# e_layers=2
# d_model=256
# d_state=2
# d_ff=256
# enc_in=7
# dec_in=7
# c_out=7
# data=ETTh1
# root_path="./dataset/ETT-small/"
# data_path="ETTh1.csv"
# features=M
# des='Exp'
# itr=1

# # 不同预测长度对应的学习率（可选）
# declare -A lr_map
# lr_map[96]=0.00007
# lr_map[192]=0.00007
# lr_map[336]=0.00005
# lr_map[720]=0.00005

# # 预测长度列表
# declare -a pred_len_list=(96 192 336 720)

# # 遍历训练
# for pred_len in "${pred_len_list[@]}"; do
#     learning_rate=${lr_map[$pred_len]}
#     model_id=ETTh1_${seq_len}_${pred_len}
#     log_file="./logs/${model_name}_${model_id}.log"
    
#     python -u run.py \
#       --is_training 1 \
#       --root_path "$root_path" \
#       --data_path "$data_path" \
#       --model_id "$model_id" \
#       --model "$model_name" \
#       --data "$data" \
#       --features "$features" \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --e_layers $e_layers \
#       --enc_in $enc_in \
#       --dec_in $dec_in \
#       --c_out $c_out \
#       --des "$des" \
#       --d_model $d_model \
#       --d_state $d_state \
#       --d_ff $d_ff \
#       --itr $itr \
#       --learning_rate $learning_rate > "$log_file" 2>&1

#     echo "✅ Training completed for ETTh1 | pred_len=$pred_len | LR=$learning_rate | Log: $log_file"
# done

#!/bin/bash

# 设置 GPU
export CUDA_VISIBLE_DEVICES=0

model_name=S_Mamba
seq_len=96
e_layers=2
d_model=256
d_ff=256
d_state=2
enc_in=7
dec_in=7
c_out=7
data=ETTh2
root_path="./dataset/ETT-small/"
data_path="ETTh2.csv"
features=M
des='Exp'
itr=1

# 不同 pred_len 对应的学习率（根据你的设置）
declare -A lr_map
lr_map[96]=0.00004
lr_map[192]=0.00004
lr_map[336]=0.00003
lr_map[720]=0.00007

# 预测长度列表
pred_lens=(96 192 336 720)

# 循环执行
for pred_len in "${pred_lens[@]}"; do
    learning_rate=${lr_map[$pred_len]}
    model_id=${data}_${seq_len}_${pred_len}
    log_file="./logs/${model_name}_${model_id}.log"

    echo "🚀 开始训练: ${model_name} | ${data} | pred_len=${pred_len} | LR=${learning_rate} | 日志: $log_file"

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
        --e_layers $e_layers \
        --enc_in $enc_in \
        --dec_in $dec_in \
        --c_out $c_out \
        --des "$des" \
        --d_model $d_model \
        --d_ff $d_ff \
        --d_state $d_state \
        --learning_rate $learning_rate \
        --itr $itr > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ 成功完成: ${model_id}"
    else
        echo "❌ 训练失败: ${model_id}，查看日志: $log_file"
    fi
    echo "---"
done

export CUDA_VISIBLE_DEVICES=0

# 创建日志目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=S_Mamba
seq_len=96
e_layers=2
enc_in=7
dec_in=7
c_out=7
data=ETTm1
root_path="./dataset/ETT-small/"
data_path="ETTm1.csv"
features=M
des='Exp'
itr=1

# 不同预测长度对应的学习率（保持高精度任务使用稍大学习率）
declare -A lr_map
lr_map[96]=0.00005
lr_map[192]=0.00005
lr_map[336]=0.00005
lr_map[720]=0.00005

# d_model 和 d_ff 根据 pred_len 条件设置（如你原始脚本中：96步用256，其余用128）
# 我们通过关联数组来配置
declare -A d_model_map
d_model_map[96]=256
d_model_map[192]=128
d_model_map[336]=128
d_model_map[720]=128

declare -A d_ff_map
d_ff_map[96]=256
d_ff_map[192]=128
d_ff_map[336]=128
d_ff_map[720]=128

# d_state 统一为 2
d_state=2

# 预测长度列表
declare -a pred_len_list=(96 192 336 720)

# 遍历训练
for pred_len in "${pred_len_list[@]}"; do
    learning_rate=${lr_map[$pred_len]}
    d_model=${d_model_map[$pred_len]}
    d_ff=${d_ff_map[$pred_len]}
    model_id=ETTm1_${seq_len}_${pred_len}
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

    echo "✅ Training completed for ETTm1 | pred_len=$pred_len | LR=$learning_rate | d_model=$d_model | Log: $log_file"
done


export CUDA_VISIBLE_DEVICES=0

# 创建日志目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=S_Mamba
seq_len=96
e_layers=2
enc_in=7
dec_in=7
c_out=7
data=ETTm2
root_path="./dataset/ETT-small/"
data_path="ETTm2.csv"
features=M
des='Exp'
itr=1
d_state=2

# 不同预测长度对应的学习率（注意：336 是 0.00003，其余为 0.00005）
declare -A lr_map
lr_map[96]=0.00005
lr_map[192]=0.00005
lr_map[336]=0.00003
lr_map[720]=0.00005

# d_model 配置：96 步用 256，其余用 128
declare -A d_model_map
d_model_map[96]=256
d_model_map[192]=128
d_model_map[336]=128
d_model_map[720]=128

# d_ff 配置：与 d_model 一致
declare -A d_ff_map
d_ff_map[96]=256
d_ff_map[192]=128
d_ff_map[336]=128
d_ff_map[720]=128

# 预测长度列表
declare -a pred_len_list=(96 192 336 720)

# 遍历训练
for pred_len in "${pred_len_list[@]}"; do
    learning_rate=${lr_map[$pred_len]}
    d_model=${d_model_map[$pred_len]}
    d_ff=${d_ff_map[$pred_len]}
    model_id=ETTm2_${seq_len}_${pred_len}
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

    echo "✅ Training completed for ETTm2 | pred_len=$pred_len | LR=$learning_rate | d_model=$d_model | Log: $log_file"
done

export CUDA_VISIBLE_DEVICES=0

# 创建日志目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=S_Mamba
seq_len=96
e_layers=3
enc_in=321
dec_in=321
c_out=321
data=custom
root_path="./dataset/electricity/"
data_path="electricity.csv"
features=M
des='Exp'
itr=1
d_model=512
d_ff=512
train_epochs=5
batch_size=16

# 不同预测长度对应的学习率
declare -A lr_map
lr_map[96]=0.001
lr_map[192]=0.0005
lr_map[336]=0.0005
lr_map[720]=0.0005

# d_state 仅在 pred_len=96 时显式设置为 16，其余情况不传参（保持原始行为）
# 因此我们通过条件判断来决定是否添加 --d_state 参数

# 预测长度列表
declare -a pred_len_list=(96 192 336 720)

# 遍历训练
for pred_len in "${pred_len_list[@]}"; do
    learning_rate=${lr_map[$pred_len]}
    model_id=ECL_${seq_len}_${pred_len}
    log_file="./logs/${model_name}_${model_id}.log"

    # 构建基础命令
    cmd="python -u run.py \
      --is_training 1 \
      --root_path \"$root_path\" \
      --data_path \"$data_path\" \
      --model_id \"$model_id\" \
      --model \"$model_name\" \
      --data \"$data\" \
      --features \"$features\" \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --enc_in $enc_in \
      --dec_in $dec_in \
      --c_out $c_out \
      --des \"$des\" \
      --d_model $d_model \
      --d_ff $d_ff \
      --train_epochs $train_epochs \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --itr $itr"

    # 只有 pred_len=96 时添加 --d_state 16
    if [ "$pred_len" -eq 96 ]; then
        cmd="$cmd --d_state 16"
    fi

    # 执行命令并重定向日志
    eval $cmd > "$log_file" 2>&1

    echo "✅ Training completed for ECL | pred_len=$pred_len | LR=$learning_rate $( [ "$pred_len" -eq 96 ] && echo "| d_state=16" ) | Log: $log_file"
done


export CUDA_VISIBLE_DEVICES=0

# 创建日志目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=S_Mamba
seq_len=96
e_layers=2
enc_in=8
dec_in=8
c_out=8
data=custom
root_path="./dataset/exchange_rate/"
data_path="exchange_rate.csv"
features=M
des='Exp'
itr=1
d_model=128
d_ff=128
batch_size=16  # 所有任务都使用 batch_size=16

# 不同预测长度对应的学习率
declare -A lr_map
lr_map[96]=0.0001
lr_map[192]=0.0001
lr_map[336]=0.00005
lr_map[720]=0.00005

# 预测长度列表
declare -a pred_len_list=(96 192 336 720)

# 遍历训练
for pred_len in "${pred_len_list[@]}"; do
    learning_rate=${lr_map[$pred_len]}
    model_id=Exchange_${seq_len}_${pred_len}
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
      --e_layers $e_layers \
      --enc_in $enc_in \
      --dec_in $dec_in \
      --c_out $c_out \
      --des "$des" \
      --d_model $d_model \
      --d_ff $d_ff \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --itr $itr > "$log_file" 2>&1

    echo "✅ Training completed for Exchange | pred_len=$pred_len | LR=$learning_rate | Log: $log_file"
done


export CUDA_VISIBLE_DEVICES=0

# 创建日志目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=S_Mamba
seq_len=96
pred_len_list=(96 192 336 720)
e_layers=2
enc_in=137
dec_in=137
c_out=137
data=Solar
root_path="./dataset/Solar/"
data_path="solar_AL.txt"
features=M
des='Exp'
itr=1
d_model=512
d_ff=512

# 固定学习率（原始脚本未指定，使用默认或模型内部默认值）
# 如果 run.py 中有默认 learning_rate，则无需设置；若需显式指定，可添加：
# --learning_rate 0.0001  # 示例值，原始脚本未提供，故不添加

# 遍历训练
for pred_len in "${pred_len_list[@]}"; do
    model_id=solar_${seq_len}_${pred_len}
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
      --e_layers $e_layers \
      --enc_in $enc_in \
      --dec_in $dec_in \
      --c_out $c_out \
      --des "$des" \
      --d_model $d_model \
      --d_ff $d_ff \
      --itr $itr > "$log_file" 2>&1

    echo "✅ Training completed for Solar | pred_len=$pred_len | Log: $log_file"
done


export CUDA_VISIBLE_DEVICES=0

# 创建日志目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=S_Mamba
seq_len=96
e_layers=4
enc_in=862
dec_in=862
c_out=862
data=custom
root_path="./dataset/traffic/"
data_path="traffic.csv"
features=M
des='Exp'
itr=1
d_model=512
d_ff=512
batch_size=16  # 所有任务均使用 batch_size=16

# 不同预测长度对应的学习率
declare -A lr_map
lr_map[96]=0.001
lr_map[192]=0.001
lr_map[336]=0.002
lr_map[720]=0.0008

# 预测长度列表
declare -a pred_len_list=(96 192 336 720)

# 遍历训练
for pred_len in "${pred_len_list[@]}"; do
    learning_rate=${lr_map[$pred_len]}
    model_id=traffic_${seq_len}_${pred_len}
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
      --e_layers $e_layers \
      --enc_in $enc_in \
      --dec_in $dec_in \
      --c_out $c_out \
      --des "$des" \
      --d_model $d_model \
      --d_ff $d_ff \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --itr $itr > "$log_file" 2>&1

    echo "✅ Training completed for Traffic | pred_len=$pred_len | LR=$learning_rate | Log: $log_file"
done

export CUDA_VISIBLE_DEVICES=0

# 创建日志目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=S_Mamba
seq_len=96
e_layers=3
enc_in=21
dec_in=21
c_out=21
data=custom
root_path="./dataset/weather/"
data_path="weather.csv"
features=M
des='Exp'
itr=1
d_model=512
d_ff=512
d_state=2
learning_rate=0.00005
train_epochs=5
batch_size=32  # 如果 run.py 有默认值则可省略；若需显式设置，请取消注释传参

# 预测长度列表
declare -a pred_len_list=(96 192 336 720)

# 遍历训练
for pred_len in "${pred_len_list[@]}"; do
    model_id=weather_${seq_len}_${pred_len}
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
      --e_layers $e_layers \
      --enc_in $enc_in \
      --dec_in $dec_in \
      --c_out $c_out \
      --des "$des" \
      --d_model $d_model \
      --d_ff $d_ff \
      --d_state $d_state \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --itr $itr > "$log_file" 2>&1

    echo "✅ Training completed for Weather | pred_len=$pred_len | LR=$learning_rate | Log: $log_file"
done