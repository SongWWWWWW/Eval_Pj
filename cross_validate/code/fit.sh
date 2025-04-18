# /bin/bash
python cross_validated_with_irt.py \
    --score_path ../data/discarded_items.jsonl \
    --output_file ../data/best_irt_gam_2pl.jsonl \
    --model_type 2pl \
    --temp_path ../temp/ \
    --n_trials 10000 \
    --parameters_path ../data/best_parameters.json



python cross_validated_with_irt.py \
    --score_path ../data/discarded_items.jsonl \
    --output_file ../data/best_irt_gam.jsonl \
    --model_type GAM \
    --temp_path ../temp/ \
    --n_trials 10000 \
    --parameters_path ../data/best_parameters.json
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