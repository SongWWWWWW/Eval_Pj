import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pygam import LinearGAM
from tqdm import tqdm

def load_data(jsonl_path):
    details = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            details.append([int(x) for x in data['detail']])
    return np.array(details)

def evaluate_subset(score_matrix, sampled_indices):
    sub_scores = score_matrix[:, sampled_indices]
    sub_total_scores = sub_scores.sum(axis=1)
    total_scores = score_matrix.sum(axis=1)

    X_train, X_val, y_train, y_val = train_test_split(
        sub_total_scores.reshape(-1, 1), total_scores, test_size=0.2, random_state=42
    )

    gam = LinearGAM().fit(X_train, y_train)
    y_pred = gam.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

def run_cross_validated_subsampling(jsonl_path, k_values=[100, 200, 500, 1000], n_trials=10000, output_file="best_subsets.jsonl"):
    score_matrix = load_data(jsonl_path)
    n_items = score_matrix.shape[1]

    best_records = []

    for k in k_values:
        print(f"\nRunning for k = {k} ...")
        best_rmse = float("inf")
        best_indices = []

        for _ in tqdm(range(n_trials)):
            sampled_indices = np.random.choice(n_items, size=k, replace=False)
            rmse = evaluate_subset(score_matrix, sampled_indices)
            if rmse < best_rmse:
                best_rmse = rmse
                best_indices = sampled_indices.tolist()

        best_records.append({
            "k": k,
            "rmse": best_rmse,
            "sampled_indices": best_indices
        })

    # 保存为 JSONL 文件
    with open(output_file, "w") as f:
        for record in best_records:
            f.write(json.dumps(record) + "\n")

    # 打印结果表格
    print("\n📊 最佳 RMSE 表格（每个 k 取最小值）：")
    print("{:<10} {:<10}".format("k", "Min RMSE"))
    print("-" * 22)
    for record in best_records:
        print("{:<10} {:.4f}".format(record["k"], record["rmse"]))

    print(f"\n✅ 结果已保存到 {output_file}")

if __name__ == "__main__":
    run_cross_validated_subsampling("../data/discarded_items.jsonl")
