for model in \
"Qwen/Qwen3-1.7B" \
"/mnt/nappipe/users/jynam/geeho/nemotron_cascade_output/Qwen3-1.7B-ifrl_ifeval/global_step_50/actor/huggingface" \
"/mnt/nappipe/users/jynam/geeho/nemotron_cascade_output/Qwen3-1.7B-math/stage2/global_step_40/actor/huggingface" \
"/mnt/nappipe/users/jynam/geeho/nemotron_cascade_output/Qwen3-1.7B-math-if/global_step_50/actor/huggingface" \
/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-naive-merge/uniform \
/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-naive-merge/data_weighted 
do
  MODEL_PATH="$model" bash /mnt/ddn/vuvlm/geeho/verl_nemotron_merge/eval_scripts/run_eval_aime25.sh
done