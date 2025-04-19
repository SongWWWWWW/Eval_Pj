# /bin/bash
# 动态生成日志文件路径
MODEL_TYPE="MLP"
N_TRIALS=1000
CURRENT_DATE=$(date +"%Y-%m-%d")  # 获取当前日期，格式为 YYYY-MM-DD
LOG_DIR="/home/tiger/Desktop/Eval_pj/Eval_Pj/cross_validate/log/${MODEL_TYPE}_n${N_TRIALS}_${CURRENT_DATE}.log"

# 确保日志目录存在
mkdir -p /home/tiger/Desktop/Eval_pj/Eval_Pj/cross_validate/log

# 执行 Python 脚本并将输出重定向到日志文件
CUDA_VISIBLE_DEVICES=0 python cross_validated_with_irt.py \
    --score_path ../data/discarded_items.jsonl \
    --output_file ../data/best_irt_gam_2pl.jsonl \
    --model_type "$MODEL_TYPE" \
    --n_trials "$N_TRIALS" \
    --parameters_path ../data/best_parameters.json \
    --num_processes 10 > "$LOG_DIR" 2>&1


conda deactivate
conda activate gui
# take the card
PYTHON_SCRIPT="/home/tiger/Desktop/monitor/llava_inf.py"

PIDS=()

for GPU_ID in {0}
do
    CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT --gpu $GPU_ID &
    PIDS+=("$!")
done

wait

# python cross_validated_with_irt.py \
#     --score_path ../data/discarded_items.jsonl \
#     --output_file ../data/best_irt_gam.jsonl \
#     --model_type GAM \
#     --temp_path ../temp/ \
#     --n_trials 10000 \
#     --parameters_path ../data/best_parameters.json
# python cross_validated_with_irt.py \
#     --score_path "../data/discarded_items.jsonl" \
#     --output_file "../best_irt_gam_3pl.jsonl" \
#     --model_type "3PL"

# python cross_validated_with_irt.py \
#     --score_path "../data/discarded_items.jsonl" \
#     --output_file "../best_irt_gam_4pl.jsonl" \
#     --model_type "4PL"

# python cross_validated_with_irt.py \
#     --score_path "../data/discarded_items.jsonl" \
#     --output_file "../best_irt_gam_.jsonl" \
#     --model_type "GAM"