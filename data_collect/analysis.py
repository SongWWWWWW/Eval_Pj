import os
import json
from collections import defaultdict
from tqdm import tqdm
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def process_file(filename, input_dir):
    """处理单个文件的工作函数"""
    if not filename.endswith(".jsonl"):
        print(f"{filename} is not a .jsonl file, skipping.")
        return None

    model_name = filename.replace(".jsonl", "")
    filepath = os.path.join(input_dir, filename)
    acc_details = []

    try:
        with open(filepath, "r") as f:
            for idx, line in enumerate(f):
                try:
                    entry = ast.literal_eval(line)
                    acc_details.append({
                        f"q_{idx}": 1 if entry["acc"]==True or entry["acc"]=='True' else 0
                    })
                except Exception as e:
                    print(f"解析错误 {filename}: {str(e)}")
                    continue
    except Exception as e:
        print(f"文件读取错误 {filename}: {str(e)}")
        return None

    if not acc_details:
        return None

    total = len(acc_details)
    acc_count = sum(1 for d in acc_details if list(d.values())[0] == 1)
    acc_rate = round(acc_count / total, 4)

    return {
        "model_name": model_name,
        "acc": acc_rate,
        "detail": acc_details
    }

def process_files(input_dir="data_collect_all", output_file="all_models_results_detailed.jsonl", max_workers=25):
    results = []
    file_list = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]

    # 使用线程池处理文件
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_file, filename, input_dir): filename
                  for filename in file_list}

        # 使用tqdm显示进度
        with tqdm(total=len(futures), desc="Processing files") as pbar:
            for future in as_completed(futures):
                filename = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {str(e)}")
                finally:
                    pbar.update(1)

    # 按模型名称排序结果
    results.sort(key=lambda x: x["acc"], reverse=True)

    # 写入汇总文件
    with open(output_file, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    # 统计分布
    stats = defaultdict(int)
    for item in results:
        rate = item["acc"]
        key = f"{int(rate*100)//10*10}%-{(int(rate*100)//10+1)*10}%"
        stats[key] += 1

    print("\n准确率分布统计:")
    for k, v in sorted(stats.items()):
        print(f"{k}: {v}个模型")

if __name__ == "__main__":
    process_files()