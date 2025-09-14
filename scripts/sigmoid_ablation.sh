#!/bin/bash
ma_type=ema
alpha=0.3
beta=0.3

current_time=$(date +"%m%d_%H%M")

seq_len_meta=96
testname=mamba_cnn_ablation_sigmoidloss
pred_lens_meta=(96 192 336 720) # 96 192 336 
epochs=(20 20 20 20)
patch_len=16
stride=8
label_len_meta=48
lradj='type3'

# === 定义每个数据集的专属参数（使用关联数组）===
declare -A batch_sizes_per_dataset
batch_sizes_per_dataset["ETTh1"]=128
batch_sizes_per_dataset["ETTh2"]=32
batch_sizes_per_dataset["ETTm1"]=256
batch_sizes_per_dataset["ETTm2"]=256
batch_sizes_per_dataset["weather"]=128
batch_sizes_per_dataset["traffic"]=32
batch_sizes_per_dataset["electricity"]=64
batch_sizes_per_dataset["exchange_rate"]=32
batch_sizes_per_dataset["national_illness"]=32
batch_sizes_per_dataset["solar"]=128

declare -A learning_rates_per_dataset
learning_rates_per_dataset["ETTh1"]=0.0005
learning_rates_per_dataset["ETTh2"]=0.0001
learning_rates_per_dataset["ETTm1"]=0.0001
learning_rates_per_dataset["ETTm2"]=0.0005
learning_rates_per_dataset["weather"]=0.0001
learning_rates_per_dataset["traffic"]=0.001
learning_rates_per_dataset["electricity"]=0.0005
learning_rates_per_dataset["exchange_rate"]=0.0001
learning_rates_per_dataset["national_illness"]=0.0005
learning_rates_per_dataset["solar"]=0.0001

declare -A enc_in_per_dataset
enc_in_per_dataset["ETTh1"]=7
enc_in_per_dataset["ETTh2"]=7  
enc_in_per_dataset["ETTm1"]=7
enc_in_per_dataset["ETTm2"]=7 
enc_in_per_dataset["weather"]=21
enc_in_per_dataset["traffic"]=862
enc_in_per_dataset["electricity"]=321
enc_in_per_dataset["exchange_rate"]=8
enc_in_per_dataset["national_illness"]=7
enc_in_per_dataset["solar"]=137

declare -A data_per_dataset
data_per_dataset["ETTh1"]=ETTh1
data_per_dataset["ETTh2"]=ETTh2
data_per_dataset["ETTm1"]=ETTm1
data_per_dataset["ETTm2"]=ETTm2
data_per_dataset["weather"]=custom
data_per_dataset["traffic"]=custom
data_per_dataset["electricity"]=custom
data_per_dataset["exchange_rate"]=custom
data_per_dataset["national_illness"]=custom
data_per_dataset["solar"]=Solar

declare -A data_path_per_dataset
data_path_per_dataset["ETTh1"]=ETTh1.csv
data_path_per_dataset["ETTh2"]=ETTh2.csv
data_path_per_dataset["ETTm1"]=ETTm1.csv
data_path_per_dataset["ETTm2"]=ETTm2.csv
data_path_per_dataset["weather"]=weather.csv
data_path_per_dataset["traffic"]=traffic.csv
data_path_per_dataset["electricity"]=electricity.csv
data_path_per_dataset["exchange_rate"]=exchange_rate.csv
data_path_per_dataset["national_illness"]=national_illness.csv
data_path_per_dataset["solar"]=solar.txt

declare -A d_state_per_dataset
d_state_per_dataset["ETTh1"]=2
d_state_per_dataset["ETTh2"]=2
d_state_per_dataset["ETTm1"]=2
d_state_per_dataset["ETTm2"]=2
d_state_per_dataset["weather"]=2
d_state_per_dataset["traffic"]=32
d_state_per_dataset["electricity"]=32
d_state_per_dataset["exchange_rate"]=32
d_state_per_dataset["national_illness"]=32
d_state_per_dataset["solar"]=32

declare -A d_model_per_dataset
d_model_per_dataset["ETTh1"]=256
d_model_per_dataset["ETTh2"]=256
d_model_per_dataset["ETTm1"]=128
d_model_per_dataset["ETTm2"]=128
d_model_per_dataset["weather"]=512
d_model_per_dataset["traffic"]=256
d_model_per_dataset["electricity"]=256
d_model_per_dataset["exchange_rate"]=128
d_model_per_dataset["national_illness"]=256
d_model_per_dataset["solar"]=256

datasets=("solar" "traffic")
#("ETTh1" "ETTh2" "ETTm1" "ETTm2" "weather" "exchange_rate" "national_illness" "electricity" "solar" "traffic")
model_name="PaDuM"
# === 定义消融实验参数范围 ===
slopes=(0.5)
centers=(30.0)
lower_bounds=(0.2)

# === 主循环 ===
for slope in "${slopes[@]}"; do
    for center in "${centers[@]}"; do
        for lb in "${lower_bounds[@]}"; do
            combo_dir="Slope_${slope}_Center_${center}_LB_${lb}"

            for dataset in "${datasets[@]}"; do
                batch_size=${batch_sizes_per_dataset[$dataset]}
                learning_rate=${learning_rates_per_dataset[$dataset]}
                enc_in=${enc_in_per_dataset[$dataset]}
                data=${data_per_dataset[$dataset]}
                data_path=${data_path_per_dataset[$dataset]}
                d_state=${d_state_per_dataset[$dataset]}
                d_model=${d_model_per_dataset[$dataset]}

                if [ "$dataset" = "national_illness" ]; then
                    seq_len=36
                    pred_lens=(24 36 48 60)
                    label_len=18
                else
                    seq_len=$seq_len_meta
                    pred_lens=("${pred_lens_meta[@]}")
                    label_len=$label_len_meta
                fi

                    log_dir="./logs/$model_name/$testname/$ma_type/$combo_dir/$dataset"
                    mkdir -p "$log_dir"

                    for i in "${!pred_lens[@]}"; do
                        pred_len=${pred_lens[$i]}
                        train_epoch=${epochs[$i]}
                        log_file="$log_dir/${model_name}_${dataset}_${seq_len}_${pred_len}_${current_time}.log"

                        echo "Running: $model_name on $dataset | lr=$learning_rate, bs=$batch_size, pred_len=$pred_len, S=$slope, C=$center, LB=$lb"
                        python -u run.py \
                            --Exp exp_main \
                            --revin 1 \
                            --is_training 1 \
                            --root_path ./dataset/ \
                            --data_path "$data_path" \
                            --model_id "${dataset}_${pred_len}_${ma_type}" \
                            --model "$model_name" \
                            --data "$data" \
                            --features M \
                            --Slope "$slope" \
                            --Center "$center" \
                            --lower_bound "$lb" \
                            --seq_len "$seq_len" \
                            --pred_len "$pred_len" \
                            --patch_len "$patch_len" \
                            --d_state "$d_state" \
                            --d_model "$d_model" \
                            --stride "$stride" \
                            --label_len "$label_len" \
                            --enc_in "$enc_in" \
                            --des 'Exp' \
                            --itr 1 \
                            --batch_size "$batch_size" \
                            --learning_rate "$learning_rate" \
                            --lradj "$lradj" \
                            --train_epochs "$train_epoch" \
                            --ma_type "$ma_type" \
                            --alpha "$alpha" \
                            --beta "$beta" > "$log_file" 2>&1
                    done
            done
        done
    done
done
# === 汇总结果 ===
for slope in "${slopes[@]}"; do
    for center in "${centers[@]}"; do
        for lb in "${lower_bounds[@]}"; do
            combo_dir="Slope_${slope}_Center_${center}_LB_${lb}"
            echo "All training tasks are completed. Generating the results table..."
            log_base_dir="./logs/$model_name/$testname/$ma_type/$combo_dir"
            output_csv_path="$log_base_dir/${model_name}_results.csv"

            # 递归遍历所有组合目录
            python generate_table.py --log_dir "$log_base_dir" --output_csv_path "$output_csv_path"
            echo "Results table has been generated and saved."

        done
    done
done
