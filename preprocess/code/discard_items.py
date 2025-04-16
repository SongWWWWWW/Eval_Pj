import argparse
import json
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

def parse_args():
    parser = argparse.ArgumentParser(description="基于题目区分度(r_pbis)和标准差过滤题目")
    parser.add_argument("--input", default="../discard/discarded_LLM_detailed.jsonl", help="输入文件路径")
    parser.add_argument("--output", default="../discard/discarded_items_detailed.jsonl", help="输出文件路径")
    parser.add_argument("--min_rpbis", type=float, default=0.05,
                       help="最小点二列相关系数阈值")
    parser.add_argument("--min_std", type=float, default=0.001,
                       help="最小标准差阈值（1%=0.01）")
    parser.add_argument("--max_acc", type=float, default=0.95,
                       help="最大允许题目准确率")
    parser.add_argument("--show_removed", type=int, default=20,
                       help="显示被过滤题目的最大数量")
    return parser.parse_args()

def load_data(args):
    """加载数据并准备计算"""
    models = []
    all_scores = []

    with open(args.input, "r") as f:
        for line in tqdm(f, desc="加载数据"):
            model = json.loads(line)
            detail_data = model["detail"]  # 原始字典列表
            answers = [int(list(d.values())[0]) for d in detail_data]
            models.append({
                "model_name": model["model_name"],
                "answers": answers,
                "original_detail": detail_data,  # 新增原始detail存储
                "total_score": sum(answers)
            })
            all_scores.append(answers)

    question_matrix = np.array(all_scores).T
    return models, question_matrix

def calculate_rpbis(question_vector, total_scores):
    """计算点二列相关系数"""
    return pearsonr(question_vector, total_scores)[0]

def filter_questions(models, question_matrix, args):
    """基于三个条件过滤题目"""
    total_scores = np.array([m["total_score"] for m in models])
    valid_questions = []
    removed_questions = {
        "high_acc": [],    # (题号, 准确率)
        "low_std": [],     # (题号, 标准差)
        "low_rpbis": []    # (题号, r_pbis值)
    }

    for q_idx in tqdm(range(question_matrix.shape[0]), desc="分析题目"):
        question_vector = question_matrix[q_idx]
        acc = np.mean(question_vector)
        std = np.std(question_vector)

        if acc > args.max_acc:
            removed_questions["high_acc"].append((q_idx, acc))
            continue

        if std < args.min_std:
            removed_questions["low_std"].append((q_idx, std))
            continue

        r_pbis = calculate_rpbis(question_vector, total_scores)
        if abs(r_pbis) < args.min_rpbis:
            removed_questions["low_rpbis"].append((q_idx, r_pbis))
            continue

        valid_questions.append(q_idx)

    return valid_questions, removed_questions

def print_removed_stats(removed_questions, args):
    """打印被过滤题目的详细信息"""
    print("\n题目过滤统计:")
    total_removed = sum(len(v) for v in removed_questions.values())
    print(f"被过滤题目总数: {total_removed}")

    for reason, items in removed_questions.items():
        if not items:
            continue

        print(f"\n=== 因【{get_reason_name(reason)}】被过滤的题目 ===")
        print(f"数量: {len(items)}")

        # 按严重程度排序
        if reason == "high_acc":
            items.sort(key=lambda x: -x[1])  # 按准确率降序
        elif reason == "low_std":
            items.sort(key=lambda x: x[1])    # 按标准差升序
        else:
            items.sort(key=lambda x: abs(x[1]))  # 按r_pbis绝对值升序

        # 显示前N个
        print(f"前{min(args.show_removed, len(items))}个题目:")
        for q_idx, value in items[:args.show_removed]:
            print(f"  题目{q_idx:4d}: {get_value_desc(reason, value)}")

def get_reason_name(reason):
    names = {
        "high_acc": "准确率过高",
        "low_std": "标准差过低",
        "low_rpbis": "区分度不足"
    }
    return names.get(reason, reason)

def get_value_desc(reason, value):
    if reason == "high_acc":
        return f"准确率={value:.1%}"
    elif reason == "low_std":
        return f"标准差={value:.4f}"
    else:
        return f"r_pbis={value:.4f}"

def save_results(models, valid_questions, args):
    """保存过滤后的结果"""
    with open(args.output, "w") as f:
        for model in tqdm(models, desc="保存结果"):
            filtered_detail = [model["original_detail"][q] for q in valid_questions]
            filtered_score = sum(model["answers"][q] for q in valid_questions)

            result = {
                "model_name": model["model_name"],
                "original_acc": model["total_score"]/len(model["answers"]),
                "filtered_acc": filtered_score/len(valid_questions),
                "num_original_questions": len(model["answers"]),
                "num_filtered_questions": len(valid_questions),
                "detail": filtered_detail
            }
            f.write(json.dumps(result) + "\n")

def main():
    args = parse_args()
    print(f"过滤条件:")
    print(f" - 题目准确率 ≤ {args.max_acc}")
    print(f" - 题目标准差 ≥ {args.min_std}")
    print(f" - |r_pbis| ≥ {args.min_rpbis}")

    # 1. 加载数据
    models, question_matrix = load_data(args)
    print(f"加载完成: {len(models)} 个模型, {question_matrix.shape[0]} 道题目")

    # 2. 过滤题目
    valid_questions, removed_questions = filter_questions(models, question_matrix, args)

    # 3. 打印统计信息
    print_removed_stats(removed_questions, args)
    print(f"\n保留题目数: {len(valid_questions)}")

    # 4. 保存结果
    if valid_questions:
        save_results(models, valid_questions, args)
        print(f"\n结果已保存到 {args.output}")

        # 显示示例
        sample = models[0]
        orig_acc = sample["total_score"]/len(sample["answers"])
        new_acc = sum(sample["answers"][q] for q in valid_questions)/len(valid_questions)
        print(f"\n示例模型对比:")
        print(f"名称: {sample['model_name']}")
        print(f"原始准确率: {orig_acc:.1%} ({sample['total_score']}/{len(sample['answers'])})")
        print(f"过滤后准确率: {new_acc:.1%} ({int(new_acc*len(valid_questions))}/{len(valid_questions)})")
    else:
        print("警告: 所有题目均被过滤！")

if __name__ == "__main__":
    main()