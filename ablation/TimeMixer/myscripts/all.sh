

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=TimeMixer

seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=32
train_epochs=20
patience=10


# 预测长度分别为 96, 192, 336, 720 的四个任务
for pred_len in 96 192 336 720; do
    model_id=ECL_${seq_len}_${pred_len}
    log_file="./logs/${model_name}_${model_id}.log"
    
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ../dataset/ \
      --data_path electricity.csv \
      --model_id $model_id \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --itr 1 \
      --d_model $d_model \
      --d_ff $d_ff \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --patience $patience \
      --down_sampling_layers $down_sampling_layers \
      --down_sampling_method avg \
      --down_sampling_window $down_sampling_window > $log_file 2>&1
done



if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=TimeMixer

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=128  # 注意：你在每个任务中使用的是 batch_size=128

# 遍历不同的 pred_len
for pred_len in 96 192 336 720; do
    model_id=ETTh1_${seq_len}_${pred_len}
    log_file="./logs/${model_name}_${model_id}.log"
    
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ../dataset/ \
      --data_path ETTh1.csv \
      --model_id $model_id \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --enc_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1 \
      --d_model $d_model \
      --d_ff $d_ff \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --patience $patience \
      --batch_size $batch_size \
      --down_sampling_layers $down_sampling_layers \
      --down_sampling_method avg \
      --down_sampling_window $down_sampling_window > $log_file 2>&1

    echo "Training finished for pred_len=$pred_len, log saved to $log_file"
done



# 创建日志目录（如果不存在）
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=TimeMixer

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32

# 遍历不同的预测长度
for pred_len in 96 192 336 720; do
    model_id=ETTh2_${seq_len}_${pred_len}
    log_file="./logs/${model_name}_${model_id}.log"
    
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ../dataset/ \
      --data_path ETTh2.csv \
      --model_id $model_id \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --enc_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1 \
      --d_model $d_model \
      --d_ff $d_ff \
      --learning_rate $learning_rate \
      --down_sampling_layers $down_sampling_layers \
      --down_sampling_method avg \
      --down_sampling_window $down_sampling_window > $log_file 2>&1

    echo "✅ Training completed for ETTh2 | pred_len=$pred_len | Log saved to $log_file"
done




# 创建日志目录（如果不存在）
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=TimeMixer

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=16
train_epochs=10      
patience=10    
# 遍历不同的预测长度
for pred_len in 96 192 336 720; do
    model_id=ETTm1_${seq_len}_${pred_len}
    log_file="./logs/${model_name}_${model_id}.log"
    
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ../dataset/ \
      --data_path ETTm1.csv \
      --model_id $model_id \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --enc_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1 \
      --d_model $d_model \
      --d_ff $d_ff \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --patience $patience \
      --down_sampling_layers $down_sampling_layers \
      --down_sampling_method avg \
      --down_sampling_window $down_sampling_window > $log_file 2>&1

    echo "✅ Training completed for ETTm1 | pred_len=$pred_len | Log saved to $log_file"
done



# 创建日志目录（如果不存在）
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=TimeMixer

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=32
batch_size=128          # 注意：你在命令中使用的是 128，所以以 128 为准
train_epochs=10         # 建议补充训练轮数
patience=10              # 早停 patience

# 遍历不同的预测长度
for pred_len in 96 192 336 720; do
    model_id=ETTm2_${seq_len}_${pred_len}
    log_file="./logs/${model_name}_${model_id}.log"
    
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ../dataset/\
      --data_path ETTm2.csv \
      --model_id $model_id \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --enc_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1 \
      --d_model $d_model \
      --d_ff $d_ff \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --patience $patience \
      --down_sampling_layers $down_sampling_layers \
      --down_sampling_method avg \
      --down_sampling_window $down_sampling_window > $log_file 2>&1

    echo "✅ Training completed for ETTm2 | pred_len=$pred_len | Log saved to $log_file"
done


# 创建日志目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=TimeMixer
data_path="exchange_rate.csv"
data_name="custom"        # 因为是自定义数据集
root_path="../dataset/"

seq_len=512
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.001
d_model=16
d_ff=32
train_epochs=10
patience=3
batch_size=32
enc_in=8
c_out=8

# 预测长度列表
for pred_len in 96 192 336 720; do
    model_id=exchange_${seq_len}_${pred_len}
    log_file="./logs/${model_name}_${model_id}.log"
    
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $root_path \
      --data_path $data_path \
      --model_id $model_id \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --enc_in $enc_in \
      --c_out $c_out \
      --des 'Exp' \
      --itr 1 \
      --d_model $d_model \
      --d_ff $d_ff \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --patience $patience \
      --down_sampling_layers $down_sampling_layers \
      --down_sampling_method avg \
      --down_sampling_window $down_sampling_window > $log_file 2>&1

    echo "✅ Training completed for exchange_rate | pred_len=$pred_len | Log saved to $log_file"
done


# 创建日志目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=TimeMixer
data_path="national_illness.csv"
data_name="custom"        # 自定义数据集
root_path="../dataset/"

seq_len=148
e_layers=2
down_sampling_layers=2
down_sampling_window=2
learning_rate=0.001
d_model=16
d_ff=32
train_epochs=10
patience=3
batch_size=32
enc_in=7
c_out=7



declare -a pred_len_list=(24 36 48 60)
declare -a model_id_suffix=(96 192 336 720)  #

# 遍历预测长度
for i in "${!pred_len_list[@]}"; do
    pred_len=${pred_len_list[i]}
    suffix=${model_id_suffix[i]}
    model_id=ili_${seq_len}_${suffix}          
    log_file="./logs/${model_name}_${model_id}.log"
    
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $root_path \
      --data_path $data_path \
      --model_id $model_id \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --enc_in $enc_in \
      --c_out $c_out \
      --des 'Exp' \
      --itr 1 \
      --d_model $d_model \
      --d_ff $d_ff \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --patience $patience \
      --down_sampling_layers $down_sampling_layers \
      --down_sampling_method avg \
      --down_sampling_window $down_sampling_window > $log_file 2>&1

    echo "✅ Training completed for national_illness | pred_len=$pred_len (task=$suffix) | Log saved to $log_file"
done



# 创建日志目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=TimeMixer
seq_len=96
down_sampling_layers=2
down_sampling_window=2
learning_rate=0.001
batch_size=32
train_epochs=10
patience=3

# 数据路径
root_path="../dataset/"
data_path="solar.txt"

# 模型参数（Solar 专用）
data_name="Solar"
enc_in=137
dec_in=137
c_out=137
e_layers=3
d_layers=1
factor=3
d_model=512
d_ff=2048
use_norm=0
channel_independence=0

# 预测长度列表
declare -a pred_len_list=(96 192 336 720)

# 遍历训练
for pred_len in "${pred_len_list[@]}"; do
    model_id=solar_${seq_len}_${pred_len}
    log_file="./logs/${model_name}_${model_id}.log"
    
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "$root_path" \
      --data_path "$data_path" \
      --model_id "$model_id" \
      --model "$model_name" \
      --data "$data_name" \
      --features M \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --d_layers $d_layers \
      --factor $factor \
      --enc_in $enc_in \
      --dec_in $dec_in \
      --c_out $c_out \
      --des 'Exp' \
      --itr 1 \
      --use_norm $use_norm \
      --d_model $d_model \
      --d_ff $d_ff \
      --channel_independence $channel_independence \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --patience $patience \
      --down_sampling_layers $down_sampling_layers \
      --down_sampling_method avg \
      --down_sampling_window $down_sampling_window > "$log_file" 2>&1

    echo "✅ Training completed for Solar | pred_len=$pred_len | Log saved to $log_file"
done


#export CUDA_VISIBLE_DEVICES=0  # 可手动启用

# 创建日志目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=TimeMixer
seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=64
batch_size=8
train_epochs=10      # 建议显式设置
patience=5           # 建议显式设置

# 数据路径
root_path="./dataset/"
data_path="traffic.csv"
data_name="custom"   # 因为是自定义数据集

# 模型结构参数
d_layers=1
factor=3
enc_in=862
dec_in=862
c_out=862

# 预测长度列表
declare -a pred_len_list=(96 192 336 720)

# 遍历训练
for pred_len in "${pred_len_list[@]}"; do
    model_id=Traffic_${seq_len}_${pred_len}
    log_file="./logs/${model_name}_${model_id}.log"
    
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "$root_path" \
      --data_path "$data_path" \
      --model_id "$model_id" \
      --model "$model_name" \
      --data "$data_name" \
      --features M \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --d_layers $d_layers \
      --factor $factor \
      --enc_in $enc_in \
      --dec_in $dec_in \
      --c_out $c_out \
      --des 'Exp' \
      --itr 1 \
      --d_model $d_model \
      --d_ff $d_ff \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --patience $patience \
      --down_sampling_layers $down_sampling_layers \
      --down_sampling_method avg \
      --down_sampling_window $down_sampling_window > "$log_file" 2>&1

    echo "✅ Training completed for traffic | pred_len=$pred_len | Log saved to $log_file"
done


# 创建日志目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=TimeMixer
seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=128
train_epochs=20
patience=10

# 数据路径
root_path="../dataset/"
data_path="weather.csv"
data_name="custom"

# 模型参数
d_layers=1
factor=3
enc_in=21
dec_in=21
c_out=21

# 预测长度列表
declare -a pred_len_list=(96 192 336 720)

# 遍历训练
for pred_len in "${pred_len_list[@]}"; do
    model_id=weather_${seq_len}_${pred_len}
    log_file="./logs/${model_name}_${model_id}.log"
    
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "$root_path" \
      --data_path "$data_path" \
      --model_id "$model_id" \
      --model "$model_name" \
      --data "$data_name" \
      --features M \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --d_layers $d_layers \
      --factor $factor \
      --enc_in $enc_in \
      --dec_in $dec_in \
      --c_out $c_out \
      --des 'Exp' \
      --itr 1 \
      --d_model $d_model \
      --d_ff $d_ff \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --patience $patience \
      --down_sampling_layers $down_sampling_layers \
      --down_sampling_method avg \
      --down_sampling_window $down_sampling_window > "$log_file" 2>&1

    echo "✅ Training completed for weather | pred_len=$pred_len | Log saved to $log_file"
done