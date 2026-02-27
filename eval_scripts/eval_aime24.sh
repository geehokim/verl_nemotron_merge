for model in \
"/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-ram-fp32/unique_if_only_scale_1_fp32" \
"/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-ram-fp32/unique_math_only_scale_1_fp32" \
"/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-ram-fp32/ram_arm_fp32" \
"/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-ram-fp32/ram_plus_arm_r_v2_r1.1_fp32" \
"/mnt/nappipe/users/jynam/geeho/nemotron_cascade_output/Qwen3-1.7B-truncated_svd-merge" 
do
  MODEL_PATH="$model" bash /mnt/ddn/vuvlm/geeho/verl_nemotron_merge/eval_scripts/run_eval_aime24.sh
done