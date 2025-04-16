from huggingface_hub import HfApi

# 初始化 Hugging Face API 客户端
api = HfApi()

# 获取所有数据集列表
available_datasets = [dataset.id for dataset in api.list_datasets()]

# 筛选包含 "open-llm" 的数据集
open_llm_datasets = [ds for ds in available_datasets if "open-llm-leaderboard-old/details" in ds.lower()]

# 打印相关数据集
print("与 open-llm 相关的数据集：")
for ds in open_llm_datasets:
    print(ds)