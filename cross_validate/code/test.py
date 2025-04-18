from py_irt.io import read_jsonlines
from py_irt.config import ModelConfig
from py_irt.models import train_model_from_data

# 1. 加载数据
data_path = "../data/discarded_items.jsonl"
data = read_jsonlines(data_path)

# 2. 构造模型配置（用 4PL 模型）
config = ModelConfig(
    model_type="4pl",
    num_items=len({d['item_id'] for d in data}),
    num_subjects=len({d['subject_id'] for d in data}),
)

# 3. 训练模型
model = train_model_from_data(data=data, config=config)

# 4. 保存模型（可选）
output_dir = "../data/test-4pl"
model.save(output_dir)
