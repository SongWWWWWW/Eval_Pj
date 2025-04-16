import argparse
import json
import numpy as np
from tqdm import tqdm

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="过滤模型")
    parser.add_argument("--input", default="../raw_data/discarded_items_detailed.jsonl", help="输入文件路径")
    parser.add_argument("--output", default="../train_data/gsm8k_train_data.jsonl", help="输出文件路径")
    return parser.parse_args()

def load_results(input_file):
    """加载JSONL格式的结果文件"""
    results = []
    with open(input_file, "r") as f:
        for line in tqdm(f, desc="加载数据"):
            data = json.loads(line)
            results.append(data)
    return results

def change(data, args):
    if not data:
        return []
    formatted_data = []
    for item in tqdm(data, desc="格式化数据"):
        model_name = item["model_name"]
        responses = {}
        for d in item["detail"]:
            q_name, score = list(d.items())[0]
            responses[q_name] = int(score)
        
        formatted_data.append({
            "subject_id": model_name,
            "responses": responses
        })
    return formatted_data

def save_results(data, output_file):
    """保存过滤后的结果"""
    with open(output_file, "w") as f:
        for item in tqdm(data, desc="保存结果"):
            f.write(json.dumps(item) + "\n")

def main():
    args = parse_args()
    
    # 加载数据
    data = load_results(args.input)
    
    # 数据格式化
    formatted_data = change(data, args)
    
    # 保存结果
    save_results(formatted_data, args.output)
    print(f"\n结果已保存到 {args.output}")

if __name__ == "__main__":
    main()