for model in \
"/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-fisher-merge/if_sparse_topk_fisher/if_delta_sparse_top_0p1pct" \
"/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-fisher-merge/if_sparse_topk_fisher/if_delta_sparse_top_1pct" \
"/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-fisher-merge/if_sparse_topk_fisher/if_delta_sparse_top_5pct" \
"/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-jwcm-v2-new/if_importance_topk_sparsified/if_top_1pct" \
"/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-jwcm-v2-new/if_importance_topk_sparsified/if_top_5pct" \
"/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-jwcm-v2-new/if_importance_topk_sparsified/if_top_10pct" 
do
  MODEL_PATH="$model" bash /mnt/ddn/vuvlm/geeho/verl_nemotron_merge/eval_scripts/run_eval_ifeval.sh
done