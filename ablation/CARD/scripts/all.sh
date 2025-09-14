
# --- Content from electricity.sh ---

# export CUDA_VISIBLE_DEVICES=0,1,2,3

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=96
model_name=CARD

root_path_name=../dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

random_seed=2023
for pred_len in 96 192 336
do
python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 321 \
  --e_layers 3 \
  --n_heads 8 \
  --d_model 128 \
  --d_ff  256 \
  --dropout 0.2\
  --fc_dropout 0.2\
  --head_dropout 0\
  --patch_len 16\
  --stride 8\
  --des 'Exp' \
  --train_epochs 100\
  --patience 100 \
  --itr 1 --batch_size 16 --dp_rank 8 --learning_rate 0.0001 --merge_size 2 \
  --lradj 'CARD'\
  --warmup_epochs 20 > logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

# batch_size 32

# # --- Content from etth1.sh ---

# # export CUDA_VISIBLE_DEVICES=0

# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi

# seq_len=96
# model_name=CARD

# root_path_name=../dataset/
# data_path_name=ETTh1.csv
# model_id_name=ETTh1
# data_name=ETTh1

# random_seed=2023
# for pred_len in 96 192 336 720
# do
#   python -u run_longExp.py \
#     --random_seed $random_seed \
#     --is_training 1 \
#     --root_path $root_path_name \
#     --data_path $data_path_name \
#     --model_id $model_id_name'_'$seq_len'_'$pred_len \
#     --model $model_name \
#     --data $data_name \
#     --features M \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --enc_in 7 \
#     --e_layers 2 \
#     --n_heads 2 \
#     --d_model 16 \
#     --d_ff 32 \
#     --dropout 0.3\
#     --fc_dropout 0.3\
#     --head_dropout 0\
#     --patch_len 16\
#     --stride 8\
#     --des 'Exp' \
#     --train_epochs 100\
#     --patience 10 \
#     --itr 1 --batch_size 128 --learning_rate 0.0001 --merge_size 2 \
#     --lradj 'CARD'\
#     --warmup_epochs 0 > logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
# done

# # --- Content from etth2.sh ---

# # export CUDA_VISIBLE_DEVICES=2

# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi

# seq_len=96
# model_name=CARD

# root_path_name=../dataset/
# data_path_name=ETTh2.csv
# model_id_name=ETTh2
# data_name=ETTh2

# random_seed=2023
# for pred_len in 96 192 336 720
# do
#     python -u run_longExp.py \
#     --random_seed $random_seed \
#     --is_training 1 \
#     --root_path $root_path_name \
#     --data_path $data_path_name \
#     --model_id $model_id_name'_'$seq_len'_'$pred_len \
#     --model $model_name \
#     --data $data_name \
#     --features M \
#     --seq_len $seq_len \
#     --label_len $seq_len \
#     --pred_len $pred_len \
#     --enc_in 7 \
#     --e_layers 2 \
#     --n_heads 2 \
#     --d_model 16 \
#     --d_ff 32 \
#     --dropout 0.3\
#     --fc_dropout 0.3\
#     --head_dropout 0\
#     --patch_len 16\
#     --stride 8\
#     --des 'Exp' \
#     --train_epochs 100\
#     --patience 10 \
#     --itr 1 --batch_size 128 --learning_rate 0.0001 --merge_size 2 \
#     --lradj 'CARD'\
#     --warmup_epochs 0 > logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
# done

# --- Content from ettm1.sh ---

# export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=96
model_name=CARD

root_path_name=../dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

random_seed=2023
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
    --e_layers 2 \
    --n_heads 2 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10 \
    --itr 1 --batch_size 128 --learning_rate 0.0001 --merge_size 2 \
    --lradj 'CARD'\
    --warmup_epochs 0 > logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

# --- Content from ettm2.sh ---

# export CUDA_VISIBLE_DEVICES=2

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=96
model_name=CARD

root_path_name=../dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

random_seed=2023
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
    --e_layers 2 \
    --n_heads 2 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10 \
    --itr 1 --batch_size 128 --learning_rate 0.0001 --merge_size 2 \
    --lradj 'CARD'\
    --warmup_epochs 0 > logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

# --- Content from exchange.sh ---

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=96
model_name=CARD

root_path_name=../dataset/
data_path_name=exchange_rate.csv
model_id_name=exchange
data_name=custom

random_seed=2023
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 8 \
      --e_layers 2 \
      --n_heads 2 \
      --d_model 16 \
      --d_ff 32 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --itr 1 --batch_size 128 --learning_rate 0.0001 --merge_size 2 \
      --lradj 'CARD'\
      --warmup_epochs 0 > logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

# --- Content from illness.sh ---

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=36 # 36 104 148
model_name=CARD

root_path_name=../dataset/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

random_seed=2023
for pred_len in 24 36 48 60
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --label_len 18 \
      --e_layers 2 \
      --n_heads 2 \
      --d_model 16 \
      --d_ff 32 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 24\
      --stride 2\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --itr 1 --batch_size 128 --learning_rate 0.0001 --merge_size 2 \
      --lradj 'CARD'\
      --warmup_epochs 0 > logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

# --- Content from solar.sh ---

# export CUDA_VISIBLE_DEVICES=7,6,5,4

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=96
model_name=CARD

root_path_name=../dataset/
data_path_name=solar.txt
model_id_name=solar
data_name=Solar

random_seed=2023
for pred_len in 96 192 336 720
do
python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 137 \
    --e_layers 2 \
    --n_heads 2 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10 \
    --itr 1 --batch_size 16 --learning_rate 0.0001 --merge_size 2 \
    --lradj 'CARD'\
    --warmup_epochs 0 > logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done


# --- Content from traffic.sh ---

# export CUDA_VISIBLE_DEVICES=0,1,2,3

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=96
model_name=CARD

root_path_name=../dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

random_seed=2023
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --e_layers 2 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
    --train_epochs 100\
      --patience 100 \
      --itr 1 --batch_size 8 --learning_rate 0.001 --merge_size 16 \
      --lradj 'CARD'\
      --warmup_epochs 20 > logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

# batch_size 24

# --- Content from weather.sh ---

# export CUDA_VISIBLE_DEVICES=7,6,5,4

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=96
model_name=CARD

root_path_name=../dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

random_seed=2023
for pred_len in 96 192 336 720
do
python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 21 \
    --e_layers 2 \
    --n_heads 2 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10 \
    --itr 1 --batch_size 128 --learning_rate 0.0001 --merge_size 2 \
    --lradj 'CARD'\
    --warmup_epochs 0 > logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
