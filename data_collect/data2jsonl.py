import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm  # 导入 tqdm 库

# 定义基础目录
base_dir = "/net/scratch/eval_pj/data"

# 查找所有包含"gsm8k"的.parquet文件
def find_parquet_files(directory):
    parquet_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.parquet') and 'gsm8k' in file.lower():
                parquet_files.append(os.path.join(root, file))
    return parquet_files

# 解析模型名称和日期
def parse_model_and_date(file_path):
    parts = Path(file_path).parts
    model_name = None
    date_str = None

    # 遍历路径部分，找到模型名称和日期
    for part in reversed(parts):  # 从后往前检查以找到最近的日期和模型名
        # 修改模型名称提取逻辑
        if 'details_' in part and '__' in part:
            try:
                # 处理包含双下划线的目录名 (如 datasets--...__Yi-34B)
                model_part = part.split('__')[-1]
                if model_part.startswith("details_"):
                    model_name = model_part.split('_', 1)[1]  # 处理 details_01-ai__Yi-34B 格式
                else:
                    model_name = model_part
                break  # 找到模型名后立即停止
            except IndexError:
                print(f"解析模型名称时发生错误: {parts}")
                continue

    # 单独遍历查找日期（避免与模型名查找冲突）
    for part in reversed(parts):
        # 增强日期格式匹配（支持 2023-12-05T03-47-25.491369 格式）
        if part.startswith("20") and "T" in part and "-" in part:
            date_str = part
            break

    return model_name, date_str

# 主函数
def main():
    # 查找所有符合条件的 .parquet 文件
    print("正在查找所有包含 'gsm8k' 的 .parquet 文件...")
    parquet_files = find_parquet_files(base_dir)

    # 如果没有找到任何文件，直接退出
    if not parquet_files:
        print("未找到符合条件的文件！")
        return

    model_to_latest_file = {}

    # 使用 tqdm 显示进度条，筛选每个模型最新的文件
    print("正在筛选每个模型的最新文件...")
    for file in tqdm(parquet_files, desc="筛选文件", unit="file"):
        model_name, date_str = parse_model_and_date(file)
        if model_name is None or date_str is None:
            print(f"跳过无效文件: {file}")
            continue
        if model_name not in model_to_latest_file or date_str > model_to_latest_file[model_name][1]:
            model_to_latest_file[model_name] = (file, date_str)

    # 对于每个模型，处理其对应的最新文件
    print("正在处理并保存数据...")
    for model_name, (file, _) in tqdm(model_to_latest_file.items(), desc="处理模型", unit="model"):
        print(f"正在处理文件: {file}")
        df = pd.read_parquet(file)
        output_file = f"/net/scratch/eval_pj/data_collect_all/{model_name}.jsonl"

        # 检查是否需要从嵌套的 metrics 中提取 acc
        if 'metrics' in df.columns:
            df['acc'] = df['metrics'].apply(lambda x: x.get('acc', None))  # 提取 metrics 中的 acc

        # 添加列名兼容性处理
        column_mapping = {
            'acc': ['acc', 'accuracy'],  # 可能的列名变体
            'example': ['example', 'problem']
        }
        selected_columns = {}
        for target_col, variants in column_mapping.items():
            for var in variants:
                if var in df.columns:
                    selected_columns[target_col] = var
                    break
            else:
                raise KeyError(f"找不到{target_col}列的替代名称，文件: {file}")

        with open(output_file, 'w', encoding='utf-8') as outfile:
            for index, row in df[list(selected_columns.values())].iterrows():
                record = {
                    "acc": row[selected_columns['acc']],
                    "example": row[selected_columns['example']]
                }
                outfile.write(f"{record}\n")
        print(f"数据已保存到: {output_file}")

if __name__ == "__main__":
    main()