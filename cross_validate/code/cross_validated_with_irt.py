import json
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from pygam import LinearGAM
from tqdm import tqdm
from typing import Tuple
from IRT import IRT
import argparse

# 加载数据
def load_data(jsonl_path):
    details = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            details.append([int(x) for x in data['detail']])
    return np.array(details)

# 分割训练集和验证集
def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    percentiles = np.percentile(y, np.linspace(0, 100, 11))  # 分10组
    y_strata = np.digitize(y, percentiles[1:-1], right=True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    train_idx, val_idx = next(sss.split(X, y_strata))
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

# 计算IRT模型的能力值（Ability）
def calculate_ability(score_matrix: np.ndarray, model_type='2PL',temp_path="../temp"):
    """
    使用IRT模型（2PL、3PL、4PL）计算每个受试者的能力值
    """
    n_items = score_matrix.shape[1]
    # abilities = np.zeros(score_matrix.shape[0])
    model = IRT(score_matrix,model_type,temp_path)
    # # 构造IRT模型
    # if model_type == '2PL':
    #     model = IRT(score_matrix, model="2PL")
    # elif model_type == '3PL':
    #     model = IRT(score_matrix, model="3PL")
    # elif model_type == '4PL':
    #     model = IRT(score_matrix, model="4PL")
    # else:
    #     raise ValueError("Unsupported model type: must be one of '2PL', '3PL', '4PL'.")

    # 拟合IRT模型并计算能力值
    model.fit()
    abilities = model.ability
    return abilities

# 评估子集
def evaluate_subset(score_matrix, sampled_indices, abilities, model_type='2PL',temp_path="../temp"):
    sub_scores = score_matrix[:, sampled_indices]
    sub_total_scores = sub_scores.sum(axis=1)
    total_scores = score_matrix.sum(axis=1)

    if model_type != 'GAM':
        # 使用IRT模型计算能力值
        if abilities is None:
            abilities = calculate_ability(score_matrix, model_type=model_type,temp_path=temp_path)

        # 将能力值与子得分结合
        X_train, X_val, y_train, y_val = train_test_split(
            np.column_stack((abilities, sub_total_scores)), total_scores, test_size=0.1, random_state=42
        )
    else:
        # 只使用子得分
        X_train, X_val, y_train, y_val = train_test_split(
            sub_total_scores.reshape(-1, 1), total_scores, test_size=0.1, random_state=42
        )

    # 使用GAM模型进行训练和预测
    gam = LinearGAM().fit(X_train, y_train)
    y_pred = gam.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

# 交叉验证子集
def run_cross_validated_subsampling(score_path, ability,k_values=[100, 200, 500, 1000], n_trials=10000, output_file="best_subsets.jsonl", model_type='2PL',temp_path="../temp"):
    score_matrix = load_data(score_path)
    n_items = score_matrix.shape[1]

    best_records = []

    for k in k_values:
        print(f"\nRunning for k = {k} ...")
        best_rmse = float("inf")
        best_indices = []

        for _ in tqdm(range(n_trials)):
            sampled_indices = np.random.choice(n_items, size=k, replace=False)
            rmse = evaluate_subset(score_matrix,  sampled_indices,ability, model_type=model_type,temp_path=temp_path)
            if rmse < best_rmse:
                best_rmse = rmse
                best_indices = sampled_indices.tolist()
        print(f"Best RMSE for k={k}: {best_rmse:.4f}")

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
def parse_args():
    parser = argparse.ArgumentParser(description="Run cross-validated subsampling with GAM and IRT models.")
    parser.add_argument('--score_path', type=str, help="Path to the input score JSONL file.")
    parser.add_argument('--k_values', type=int, required=False, nargs='+', default=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500], help="List of k values to test.")
    parser.add_argument('--n_trials', type=int, required=False, default=10000, help="Number of trials to run for each k value.")
    parser.add_argument('--output_file', type=str, default="best_subsets.jsonl", help="Output file to save the best subsets.")
    parser.add_argument('--model_type', type=str, choices=['2pl', '3pl', '4pl', 'GAM'], default='2PL', help="IRT model type or 'GAM' for GAM model only.")
    parser.add_argument('--temp_path',type=str)
    parser.add_argument('--parameters_path',type=str)
    return parser.parse_args()

if __name__ == "__main__":


    args = parse_args()
    print(args)
    with open (args.parameters_path,"r") as f:
        abilities = json.load(f)
    abilities = abilities["ability"]
    run_cross_validated_subsampling(
        ability = abilities,
        score_path=args.score_path,
        k_values=args.k_values,
        n_trials=args.n_trials,
        output_file=args.output_file,
        model_type=args.model_type,
        temp_path = args.temp_path
    )


# import numpy as np
# import pandas as pd
# import json
# from tqdm import tqdm
# from scipy.special import expit  # sigmoid
# from scipy.stats import spearmanr
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import StratifiedShuffleSplit
# from pygam import LinearGAM, s
# from typing import Tuple

# # ---------- 数据加载 ----------
# def load_data(jsonl_path):
#     details = []
#     with open(jsonl_path, 'r') as f:
#         for line in f:
#             data = json.loads(line)
#             details.append([int(x) for x in data['detail']])
#     return np.array(details)

# def get_paramaters(data_path, parameters_path):
#     scores_matrix = load_data(data_path)
#     with open(parameters_path, 'r') as f:
#         parameters = json.load(f)
#     theta = np.array(parameters['ability'])
#     delta = np.array(parameters['diff'])
#     a = np.array(parameters['disc'])
#     c = np.array(parameters.get('c', np.zeros_like(a)))
#     d = np.array(parameters.get('d', np.ones_like(a)))
#     return scores_matrix, theta, delta, a, c, d

# # ---------- IRT模型 ----------
# def irt_2pl(theta, a, delta):
#     return expit(np.outer(theta, a) - delta)

# def irt_3pl(theta, a, delta, c):
#     prob = irt_2pl(theta, a, delta)
#     return c + (1 - c) * prob

# def irt_4pl(theta, a, delta, c, d):
#     prob = irt_2pl(theta, a, delta)
#     return c + (d - c) * prob

# # ---------- 预测总分 ----------
# def predict_scores(prob_matrix):
#     return prob_matrix.sum(axis=1)

# # ---------- GAM回归 ----------
# def train_and_eval_gam(y_true, features):
#     # sub_scores = features[:, sampled_indices]
#     # sub_total_scores = sub_scores.sum(axis=1)
#     # total_scores = score_matrix.sum(axis=1)
#     X_train, X_val, y_train, y_val = train_test_split(features.reshape(-1, 1), y_true, test_size=0.1, random_state=42)

#     gam = LinearGAM(s(0)).fit(X_train, y_train)
#     y_pred = gam.predict(X_val)
#     rmse = np.sqrt(mean_squared_error(y_val, y_pred))
#     spearman = spearmanr(y_val, y_pred).correlation
#     return rmse, spearman

# # ---------- 分层划分 ----------
# def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     percentiles = np.percentile(y, np.linspace(0, 100, 11))
#     y_strata = np.digitize(y, percentiles[1:-1], right=True)
#     sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
#     train_idx, val_idx = next(sss.split(X, y_strata))
#     return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

# def perturb_param(param, scale=0.05, lower=None, upper=None):
#     perturbed = param + np.random.normal(loc=0, scale=scale, size=param.shape)
#     if lower is not None:
#         perturbed = np.clip(perturbed, lower, upper)
#     return perturbed


# # ---------- 主流程 ----------
# def repeated_evaluation(score_path, parameters_path, n_trials=1000, output_path="best_irt_results.jsonl"):
#     scores_matrix, theta, delta, a, c, d = get_paramaters(score_path, parameters_path)
#     n_items = scores_matrix.shape[1]
#     true_scores = scores_matrix.sum(axis=1)

#     best_records = []

#     for model_type in ['2PL', '3PL', '4PL']:
#         print(f"\n🔍 Evaluating {model_type} for {n_trials} trials...")
#         best_rmse = float('inf')
#         best_spearman = -1
#         best_trial = -1

#         for trial in tqdm(range(n_trials)):
#             # add noise
#             # theta_perturb = perturb_param(theta, scale=0.1)
#             # a_perturb = perturb_param(a, scale=0.05, lower=0.01, upper=3.0)
#             # delta_perturb = perturb_param(delta, scale=0.1)
#             # c_perturb = perturb_param(c, scale=0.02, lower=0.0, upper=0.35)
#             # d_perturb = perturb_param(d, scale=0.02, lower=0.9, upper=1.0)
#             if model_type == '2PL':
#                 prob_matrix = irt_2pl(theta, a, delta)
#             elif model_type == '3PL':
#                 prob_matrix = irt_3pl(theta, a, delta, c)
#             elif model_type == '4PL':
#                 prob_matrix = irt_4pl(theta, a, delta, c, d)

#             pred_scores = predict_scores(prob_matrix)
#             try:
                
#                 rmse, spearman = train_and_eval_gam(true_scores, pred_scores)
#                 if rmse < best_rmse:
#                     best_rmse = rmse
#                     best_spearman = spearman
#                     best_trial = trial
#             except Exception as e:
#                 print(e)
#                 continue  # 某些样本可能会导致GAM失败，跳过即可

#         print(f"{model_type} → Best RMSE: {best_rmse:.4f}, Spearman: {best_spearman:.4f} at trial #{best_trial}")
#         best_records.append({
#             "model_type": model_type,
#             "rmse": best_rmse,
#             "spearman": best_spearman,
#             "best_trial": best_trial
#         })

#     # 保存结果
#     with open(output_path, "w") as f:
#         for record in best_records:
#             f.write(json.dumps(record) + "\n")

#     print(f"\n✅ 所有模型评估完毕，结果已保存至 {output_path}")

# if __name__ == "__main__":
#     repeated_evaluation(
#         score_path="../data/discarded_items.jsonl",
#         parameters_path="../data/best_parameters.json",
#         n_trials=1000,
#         output_path="../data/best_irt_gam_results.jsonl"
#     )
