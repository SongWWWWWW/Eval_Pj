import argparse
import json
import numpy as np
from tqdm import tqdm

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="过滤模型")
    parser.add_argument("--input", default="../model_data/model_data_detailed.jsonl", help="输入文件路径")
    parser.add_argument("--output", default="../discard/discarded_model_data_detailed.jsonl", help="输出文件路径")
    parser.add_argument("--low_percentile", type=float, default=0.1,
                       help="去除最低准确率的百分比阈值")
    parser.add_argument("--max_acc", type=float, default=0.95,
                       help="最大允许准确率阈值")
    return parser.parse_args()

def load_results(input_file):
    """加载JSONL格式的结果文件"""
    results = []
    with open(input_file, "r") as f:
        for line in tqdm(f, desc="加载数据"):
            data = json.loads(line)
            results.append(data)
    return results

def filter_models(models, args):
    """
    应用过滤规则：
    1. 去除最低准确率的模型（基于--low_percentile）
    2. 去除准确率>max_acc的模型
    """
    if not models:
        return []

    # 去除最低准确率的模型，去除准确率>max_acc的模型
    accuracies = [m["acc"] for m in models]
    cutoff = np.percentile(accuracies, args.low_percentile)
    print(f"最低{args.low_percentile}%对应的准确率{cutoff}")
    final_models = [m for m in models if m["acc"] > cutoff and m["acc"] <= args.max_acc]
    print(f"去除最低{args.low_percentile}%准确率(<{cutoff:.4f})模型和准确率>{args.max_acc}的模型后: {len(final_models)}/{len(models)}")

    return final_models

def save_results(models, output_file):
    """保存过滤后的结果"""
    with open(output_file, "w") as f:
        for model in tqdm(models, desc="保存结果"):
            output_item = {
                "model_name": model["model_name"],
                "acc": model["acc"],
                "detail": model["detail"]
            }
            f.write(json.dumps(output_item) + "\n")

def main():
    args = parse_args()
    print(f"参数设置: 输入={args.input}, 输出={args.output}")
    print(f"过滤阈值: 最低百分位={args.low_percentile}%, 最大准确率={args.max_acc}")

    # 加载数据
    models = load_results(args.input)
    print(f"加载完成，共 {len(models)} 个模型")

    # 过滤模型
    filtered_models = filter_models(models, args)

    # 保存结果
    if filtered_models:
        save_results(filtered_models, args.output)
        print(f"\n结果已保存到 {args.output}，共 {len(filtered_models)} 个模型")
    else:
        print("\n所有模型均被过滤，没有结果保存")

if __name__ == "__main__":
    main()