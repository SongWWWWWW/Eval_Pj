import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from scipy.special import expit  # sigmoid
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from pygam import LinearGAM, s
from typing import Tuple

# ---------- 数据加载 ----------
def load_data(jsonl_path):
    details = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            details.append([int(x) for x in data['detail']])
    return np.array(details)

def get_paramaters(data_path, parameters_path):
    scores_matrix = load_data(data_path)
    with open(parameters_path, 'r') as f:
        parameters = json.load(f)
    theta = np.array(parameters['ability'])
    delta = np.array(parameters['diff'])
    a = np.array(parameters['disc'])
    c = np.array(parameters.get('c', np.zeros_like(a)))
    d = np.array(parameters.get('d', np.ones_like(a)))
    return scores_matrix, theta, delta, a, c, d

# ---------- IRT模型 ----------
def irt_2pl(theta, a, delta):
    return expit(np.outer(theta, a) - delta)

def irt_3pl(theta, a, delta, c):
    prob = irt_2pl(theta, a, delta)
    return c + (1 - c) * prob

def irt_4pl(theta, a, delta, c, d):
    prob = irt_2pl(theta, a, delta)
    return c + (d - c) * prob

# ---------- 预测总分 ----------
def predict_scores(prob_matrix):
    return prob_matrix.sum(axis=1)

# ---------- GAM回归 ----------
def train_and_eval_gam(y_true, features):
    # sub_scores = features[:, sampled_indices]
    # sub_total_scores = sub_scores.sum(axis=1)
    # total_scores = score_matrix.sum(axis=1)
    X_train, X_val, y_train, y_val = train_test_split(features.reshape(-1, 1), y_true, test_size=0.1, random_state=42)

    gam = LinearGAM(s(0)).fit(X_train, y_train)
    y_pred = gam.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    spearman = spearmanr(y_val, y_pred).correlation
    return rmse, spearman

# ---------- 分层划分 ----------
def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    percentiles = np.percentile(y, np.linspace(0, 100, 11))
    y_strata = np.digitize(y, percentiles[1:-1], right=True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(sss.split(X, y_strata))
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]
def perturb_param(param, scale=0.05, lower=None, upper=None):
    perturbed = param + np.random.normal(loc=0, scale=scale, size=param.shape)
    if lower is not None:
        perturbed = np.clip(perturbed, lower, upper)
    return perturbed


# ---------- 主流程 ----------
def repeated_evaluation(score_path, parameters_path, n_trials=1000, output_path="best_irt_results.jsonl"):
    scores_matrix, theta, delta, a, c, d = get_paramaters(score_path, parameters_path)
    n_items = scores_matrix.shape[1]
    true_scores = scores_matrix.sum(axis=1)

    best_records = []

    for model_type in ['2PL', '3PL', '4PL']:
        print(f"\n🔍 Evaluating {model_type} for {n_trials} trials...")
        best_rmse = float('inf')
        best_spearman = -1
        best_trial = -1

        for trial in tqdm(range(n_trials)):
            # add noise
            # theta_perturb = perturb_param(theta, scale=0.1)
            # a_perturb = perturb_param(a, scale=0.05, lower=0.01, upper=3.0)
            # delta_perturb = perturb_param(delta, scale=0.1)
            # c_perturb = perturb_param(c, scale=0.02, lower=0.0, upper=0.35)
            # d_perturb = perturb_param(d, scale=0.02, lower=0.9, upper=1.0)
            if model_type == '2PL':
                prob_matrix = irt_2pl(theta, a, delta)
            elif model_type == '3PL':
                prob_matrix = irt_3pl(theta, a, delta, c)
            elif model_type == '4PL':
                prob_matrix = irt_4pl(theta, a, delta, c, d)

            pred_scores = predict_scores(prob_matrix)
            try:
                
                rmse, spearman = train_and_eval_gam(true_scores, pred_scores)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_spearman = spearman
                    best_trial = trial
            except Exception as e:
                print(e)
                continue  # 某些样本可能会导致GAM失败，跳过即可

        print(f"{model_type} → Best RMSE: {best_rmse:.4f}, Spearman: {best_spearman:.4f} at trial #{best_trial}")
        best_records.append({
            "model_type": model_type,
            "rmse": best_rmse,
            "spearman": best_spearman,
            "best_trial": best_trial
        })

    # 保存结果
    with open(output_path, "w") as f:
        for record in best_records:
            f.write(json.dumps(record) + "\n")

    print(f"\n✅ 所有模型评估完毕，结果已保存至 {output_path}")

if __name__ == "__main__":
    repeated_evaluation(
        score_path="../data/discarded_items.jsonl",
        parameters_path="../data/best_parameters.json",
        n_trials=1000,
        output_path="../data/best_irt_gam_results.jsonl"
    )
