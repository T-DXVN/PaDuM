import os
import re
import pandas as pd
import argparse

def parse_log_files(log_dir):
    """解析日志目录下的所有 .log 文件"""
    results = []
    for dataset_name in os.listdir(log_dir):
        dataset_path = os.path.join(log_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        for file in os.listdir(dataset_path):
            if not file.endswith('.log'):
                continue

            log_file = os.path.join(dataset_path, file)
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    last_line = lines[-1].strip() if lines else ""

                # 提取 mse 和 mae
                mse_match = re.search(r"mse:\s*([0-9.eE+-]+)", last_line)
                mae_match = re.search(r"mae:\s*([0-9.eE+-]+)", last_line)

                if not mse_match or not mae_match:
                    print(f"⚠️ 未在 {file} 中找到 mse 或 mae 值")
                    continue

                mse = round(float(mse_match.group(1)), 3)
                mae = round(float(mae_match.group(1)), 3)

                # 解析文件名
                stem = os.path.splitext(file)[0]
                parts = stem.split('_')
                if len(parts) < 4:
                    print(f"⚠️ 文件名格式不符: {file}")
                    continue

                model_name = parts[0]
                pred_len = int(parts[-3])  # 倒数第三个字段

                results.append({
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'Pred_len': pred_len,
                    'MSE': mse,
                    'MAE': mae
                })

            except Exception as e:
                print(f"❌ 处理文件 {file} 时出错: {e}")

    return results

def create_and_display_table_from_logs(log_dir, output_csv_path=None):
    """
    解析日志文件，生成格式化表格，并保存为 CSV 文件。
    
    Args:
        log_dir (str): 存放日志文件的根目录。
        output_csv_path (str, optional): CSV 文件的保存路径。如果为 None，则不保存。
    """
    raw_data = parse_log_files(log_dir)

    if not raw_data:
        print("未找到任何符合条件的日志数据。请检查 log_dir 是否正确。")
        return

    df = pd.DataFrame(raw_data)
    unique_models = df['Model'].unique()
    if len(unique_models) > 1:
        print(f"⚠️ 警告: 发现多个模型 {unique_models}，但此脚本仅支持一个模型。表格将只显示第一个模型的数据。")
        model_name = unique_models[0]
        df = df[df['Model'] == model_name]
    elif len(unique_models) == 1:
        model_name = unique_models[0]
    else:
        print("未找到任何模型名称。")
        return

    df = df.sort_values(by=['Dataset', 'Pred_len'])

    all_rows = []
    
    for dataset, group in df.groupby('Dataset'):
        for _, row in group.iterrows():
            all_rows.append({
                'Dataset': dataset,
                'Metric': row['Pred_len'],
                'MSE': row['MSE'],
                'MAE': row['MAE']
            })
        
        avg_mse = round(group['MSE'].mean(), 3)
        avg_mae = round(group['MAE'].mean(), 3)
        all_rows.append({
            'Dataset': dataset,
            'Metric': 'Avg',
            'MSE': avg_mse,
            'MAE': avg_mae
        })

    final_df = pd.DataFrame(all_rows)

    final_df = final_df.set_index(['Dataset', 'Metric'])
    final_df.columns = pd.MultiIndex.from_product([[model_name], ['MSE', 'MAE']])
    pd.options.display.float_format = '{:.3f}'.format

    print(final_df.to_string())

    if output_csv_path:
        df_to_save = final_df.copy()
        df_to_save.reset_index(inplace=True)
        df_to_save.columns = ['Models', 'Metric', 'MSE', 'MAE']
        df_to_save.to_csv(output_csv_path, index=False, float_format='%.3f')
        print(f"\n表格已成功保存到 {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a formatted table from log files.")
    parser.add_argument('--log_dir', type=str, required=True, help="The root directory of the log files.")
    parser.add_argument('--output_csv_path', type=str, default=None, help="Optional path to save the CSV file.")
    
    args = parser.parse_args()
    
    create_and_display_table_from_logs(args.log_dir, args.output_csv_path)