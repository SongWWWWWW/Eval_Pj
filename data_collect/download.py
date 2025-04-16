from huggingface_hub import HfApi, hf_hub_download

# 初始化 Hugging Face API 客户端
api = HfApi()

# 读取 data.txt 文件中的数据集标识符
data_file_path = "data.txt"  # 确保该文件路径正确
with open(data_file_path, "r", encoding="utf-8") as file:
    dataset_ids = [line.strip() for line in file if line.strip()]

# 遍历每个数据集标识符并下载文件
for dataset_id in dataset_ids:
    try:
        print(f"\n正在处理数据集: {dataset_id}")

        # 列出数据集中的所有文件
        print("正在列出数据集中的文件...")
        file_list = api.list_repo_files(repo_id=dataset_id, repo_type="dataset")

        # 如果需要筛选特定文件类型或名称，可以在这里添加过滤逻辑
        # 示例：筛选包含 "gsm8k" 的文件
        target_files = [f for f in file_list if "gsm8k" in f.lower()]

        if not target_files:
            print(f"未找到与 gsm8k 相关的文件！跳过数据集: {dataset_id}")
            continue

        # 下载目标文件
        for file_path in target_files:
            print(f"正在下载文件: {file_path}")
            local_path = hf_hub_download(
                repo_id=dataset_id,
                filename=file_path,
                repo_type="dataset",
                cache_dir="/net/scratch/eval_pj/data"  # 修改为你希望保存的目录
            )
            print(f"文件已下载到: {local_path}")

    except Exception as e:
        print(f"处理数据集 {dataset_id} 时发生错误: {e}")
        continue

print("\n所有数据集处理完成！")