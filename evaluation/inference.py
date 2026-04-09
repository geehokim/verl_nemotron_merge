from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from arguments import get_args
from tqdm import tqdm
import torch
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_vllm_model(args):
    """Load a vLLM model with specified configuration.
    
    Args:
        args: Command-line arguments containing model configuration:
            - model_folder: Directory containing the model
            - model_name: Name of the model to load
            - tokenizer_folder: Directory containing the tokenizer
            - tokenizer_name: Name of the tokenizer to load
            - tensor_parallel_size: Number of GPUs for tensor parallelism
            - yarn_factor: Scaling factor for YaRN (Yet another RoPE extensioN method)
            - max_output_len: Maximum output length
            - seed: Random seed for reproducibility
    
    Returns:
        LLM: Initialized vLLM model instance
    """
    tokenizer_path = os.path.join(args.tokenizer_folder, args.tokenizer_name)
    model_path = os.path.join(args.model_folder, args.model_name)
    tensor_parallel_size = args.tensor_parallel_size

    eager_mode = True if "DeepSeek-R1" in model_path else False
    print("eager_mode:", eager_mode)
    print("load tokenizer from %s" % tokenizer_path)
    print("load model from %s" % model_path)
    print("tensor_parallel_size:", tensor_parallel_size)

    if args.yarn_factor == 1:
        rope_scaling = None
    else:
        rope_scaling = {"rope_type":"yarn",
                        "factor": args.yarn_factor,
                        "original_max_position_embeddings":32768,
                        "attention_factor": 0.8782488562869419}

    max_output_len = int(args.max_output_len * args.yarn_factor)

    model_vllm = LLM(model_path, tokenizer=tokenizer_path, max_model_len=max_output_len,
                     trust_remote_code=True, tensor_parallel_size=tensor_parallel_size,
                     enforce_eager=eager_mode, seed=args.seed,
                     rope_scaling=rope_scaling
                     )

    return model_vllm


def apply_template(prompt, tokenizer, think=True):
    """Apply chat template to format the prompt for model input.
    
    Args:
        prompt: Either a string containing a single user message, or a list of chat messages
                with 'role' and 'content' fields
        tokenizer: HuggingFace tokenizer with chat template support
        think: Whether to enable thinking mode (default: True)
    
    Returns:
        str: Formatted prompt string ready for model input
        
    Raises:
        ValueError: If prompt is neither a string nor a list
    """
    if isinstance(prompt, str):
        chat = [
            {"role": "user", "content": prompt},
        ]
    elif isinstance(prompt, list):
        chat = prompt
    else:
        raise ValueError("prompt must be str or list")
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=think)


def get_prompt_list(args):
    """Load and preprocess prompts from the specified evaluation dataset.
    
    This function supports multiple benchmark datasets including:
    - Math: MATH, MATH500, GSM8K, Minerva Math, OmniMath, AIME
    - Coding: MBPP, HumanEval, LiveCodeBench
    - Multiple Choice: MMLU, MMLU Pro, GPQA
    - Instruction Following: IFEval, IFBench, MT-Bench
    - General: AlpacaEval, Arena-Hard
    
    Args:
        args: Command-line arguments containing:
            - eval_dataset: Name of the evaluation dataset
            - benchmark_folder: Root directory containing benchmark data
            - start_idx: Starting index for subsetting (optional)
            - end_idx: Ending index for subsetting (optional)
            - Various dataset-specific paths
    
    Returns:
        tuple: (prompt_list, qid_list)
            - prompt_list: List of formatted prompts ready for inference
            - qid_list: List of question IDs (None for some datasets)
    
    Raises:
        ValueError: If eval_dataset is not recognized
    """
    if args.eval_dataset == "mbpp":
        from data.benchmark import preprocess_mbpp_chatml_template
        input_datapath = os.path.join(args.benchmark_folder, args.mbpp_path)
        prompt_list, qid_list = preprocess_mbpp_chatml_template(input_datapath)

    elif args.eval_dataset == "mbpp_sanitized":
        from data.benchmark import preprocess_mbpp_chatml_template
        input_datapath = os.path.join(args.benchmark_folder, args.mbpp_sanitized_path)
        prompt_list, qid_list = preprocess_mbpp_chatml_template(input_datapath)

    elif args.eval_dataset == "mbpp_plus":
        from data.benchmark import preprocess_mbpp_chatml_template
        input_datapath = os.path.join(args.benchmark_folder, args.mbpp_plus_path)
        prompt_list, qid_list = preprocess_mbpp_chatml_template(input_datapath)

    elif args.eval_dataset == "math":
        from data.benchmark import preprocess_math_zeroshot_chatml_template

        input_datapath = os.path.join(args.benchmark_folder, args.math_path)
        prompt_list = preprocess_math_zeroshot_chatml_template(input_datapath)
        qid_list = None

    elif args.eval_dataset == "math500":
        from data.benchmark import preprocess_math500_zeroshot_chatml_template

        input_datapath = os.path.join(args.benchmark_folder, args.math500_path)
        prompt_list = preprocess_math500_zeroshot_chatml_template(input_datapath, use_r1=args.use_r1)
        qid_list = None

    elif args.eval_dataset == "gsm8k":
        from data.benchmark import preprocess_gsm8k_zeroshot_raw

        input_datapath = os.path.join(args.benchmark_folder, args.gsm8k_path)
        prompt_list = preprocess_gsm8k_zeroshot_raw(input_datapath)
        qid_list = None

    elif args.eval_dataset == "humaneval":
        from data.benchmark import preprocess_humaneval_raw
        input_datapath = os.path.join(args.benchmark_folder, args.humaneval_path)
        prompt_list, qid_list = preprocess_humaneval_raw(input_datapath)

    elif args.eval_dataset == "mmlu":
        from data.benchmark import preprocess_mmlu_raw_template
        input_datapath = os.path.join(args.benchmark_folder, args.mmlu_path)
        prompt_list = preprocess_mmlu_raw_template(input_datapath)
        qid_list = None

    elif args.eval_dataset == "mmlu_r1":
        from data.benchmark import preprocess_mmlu_r1_raw_template_wdai
        input_datapath = os.path.join(args.benchmark_folder, args.mmlu_path)
        prompt_list = preprocess_mmlu_r1_raw_template_wdai(input_datapath)
        qid_list = None

    elif args.eval_dataset == "alpaca_eval":
        from data.benchmark import preprocess_alpaca_eval_raw
        input_datapath = os.path.join(args.benchmark_folder, args.alpaca_eval_path)
        prompt_list, qid_list = preprocess_alpaca_eval_raw(input_datapath)

    elif args.eval_dataset == "arena_hard":
        from data.benchmark import preprocess_arena_hard_raw
        input_datapath = os.path.join(args.benchmark_folder, args.arena_hard_path)
        prompt_list, qid_list = preprocess_arena_hard_raw(input_datapath)

    elif args.eval_dataset == "arena_hard_v2":
        from data.benchmark import preprocess_arena_hard_v2_raw
        input_datapath = os.path.join(args.benchmark_folder, args.arena_hard_v2_path)
        prompt_list, qid_list = preprocess_arena_hard_v2_raw(input_datapath)

    elif args.eval_dataset == "ifeval":
        from data.benchmark import preprocess_ifeval_raw
        input_datapath = os.path.join(args.benchmark_folder, args.ifeval_path)
        prompt_list, qid_list = preprocess_ifeval_raw(input_datapath)

    elif args.eval_dataset == "ifeval_training":
        from data.benchmark import preprocess_ifeval_raw
        input_datapath = os.path.join(args.benchmark_folder, args.ifeval_training_path)
        prompt_list, qid_list = preprocess_ifeval_raw(input_datapath)

    elif args.eval_dataset == "ifbench":
        from data.benchmark import preprocess_ifbench_raw
        input_datapath = os.path.join(args.benchmark_folder, args.ifbench_path)
        prompt_list, qid_list = preprocess_ifbench_raw(input_datapath)

    elif args.eval_dataset == "mtbench_firstturn":
        from data.benchmark import preprocess_mtbench_firstturn_raw
        input_datapath = os.path.join(args.benchmark_folder, args.mtbench_path)
        prompt_list, qid_list = preprocess_mtbench_firstturn_raw(input_datapath)

    elif args.eval_dataset == "mtbench_secondturn":
        from data.benchmark import preprocess_mtbench_secondturn_raw
        input_datapath = os.path.join(args.benchmark_folder, args.mtbench_path)
        prompt_list, qid_list = preprocess_mtbench_secondturn_raw(input_datapath, args.model_output_path)

    elif args.eval_dataset == "lcb5_2408_2502":
        from data.benchmark import preprocess_livecodebench_raw
        input_datapath = os.path.join(args.benchmark_folder, args.livecodebench_path)
        prompt_list, qid_list = preprocess_livecodebench_raw(input_datapath)

    elif args.eval_dataset == "lcb6_2502_2505":
        from data.benchmark import preprocess_livecodebench_raw
        print(args)
        input_datapath = os.path.join(args.benchmark_folder, args.livecodebench6_path)
        prompt_list, qid_list = preprocess_livecodebench_raw(input_datapath)

    elif args.eval_dataset == "minerva_math":
        from data.benchmark import preprocess_minerva_math_chatml_template

        input_datapath = os.path.join(args.benchmark_folder, args.minervamath_path)
        prompt_list = preprocess_minerva_math_chatml_template(input_datapath)
        qid_list = None

    elif args.eval_dataset == "gaokao2023en":
        from data.benchmark import preprocess_gaokao2023en_chatml_template

        input_datapath = os.path.join(args.benchmark_folder, args.gaokao2023en_path)
        prompt_list = preprocess_gaokao2023en_chatml_template(input_datapath)
        qid_list = None

    elif args.eval_dataset == "olympiadbench":
        from data.benchmark import preprocess_olympiadbench_chatml_template

        input_datapath = os.path.join(args.benchmark_folder, args.olympiadbench_path)
        prompt_list = preprocess_olympiadbench_chatml_template(input_datapath)
        qid_list = None

    elif args.eval_dataset == "collegemath":
        from data.benchmark import preprocess_collegemath_chatml_template

        input_datapath = os.path.join(args.benchmark_folder, args.collegemath_path)
        prompt_list = preprocess_collegemath_chatml_template(input_datapath)
        qid_list = None

    elif args.eval_dataset == "mmlu_stem":
        from data.benchmark import preprocess_mmlu_stem_chatml_template
        input_datapath = os.path.join(args.benchmark_folder, args.mmlustem_path)
        prompt_list = preprocess_mmlu_stem_chatml_template(input_datapath)
        qid_list = None

    elif args.eval_dataset == "amc23":
        from data.benchmark import preprocess_amc23_chatml_template

        input_datapath = os.path.join(args.benchmark_folder, args.amc23_path)
        prompt_list = preprocess_amc23_chatml_template(input_datapath)
        qid_list = None

    elif args.eval_dataset == "aime24":
        from data.benchmark import preprocess_aime24_raw

        input_datapath = os.path.join(args.benchmark_folder, args.aime24_path)
        prompt_list, qid_list = preprocess_aime24_raw(input_datapath)

    elif args.eval_dataset == "aime25":
        from data.benchmark import preprocess_aime25_raw

        input_datapath = os.path.join(args.benchmark_folder, args.aime25_path)
        prompt_list, qid_list = preprocess_aime25_raw(input_datapath)

    elif args.eval_dataset == "omnimath":
        from data.benchmark import preprocess_omnimath_chatml_template
        input_datapath = os.path.join(args.benchmark_folder, args.omnimath_path)
        prompt_list = preprocess_omnimath_chatml_template(input_datapath)
        qid_list = None

    elif args.eval_dataset == "gpqa_diamond":
        from data.benchmark import preprocess_gpqa_raw_template
        input_datapath = os.path.join(args.benchmark_folder, args.gpqa_diamond_path)
        prompt_list = preprocess_gpqa_raw_template(input_datapath, use_r1=args.use_r1)
        qid_list = None

    elif args.eval_dataset == "mmlu_pro":
        from data.benchmark import preprocess_mmlu_pro_zero_shot_raw_template
        input_datapath = os.path.join(args.benchmark_folder, args.mmlupro_path)
        fewshot_datapath = os.path.join(args.benchmark_folder, args.mmlupro_fewshot_path)

        prompt_list = preprocess_mmlu_pro_zero_shot_raw_template(input_datapath, fewshot_datapath)
        qid_list = None

    else:
        raise ValueError("please input a correct eval_dataset name!")

    print("number of total prompt_list:", len(prompt_list))
    if args.start_idx != -1 and args.end_idx != -1:
        print("getting data from %d to %d" % (args.start_idx, args.end_idx))
        prompt_list = prompt_list[args.start_idx:args.end_idx]
        if qid_list:
            qid_list = qid_list[args.start_idx:args.end_idx]

    print("number of test samples in the dataset:", len(prompt_list))

    return prompt_list, qid_list


def main():
    """Main function to run inference on evaluation benchmarks.
    
    This function:
    1. Parses command-line arguments
    2. Loads the vLLM model and tokenizer
    3. Loads test data from the specified benchmark
    4. Runs batched inference with specified sampling parameters
    5. Post-processes outputs (extracts reasoning, handles special tokens)
    6. Saves results to JSONL format
    
    The output directory structure is:
        {model_folder}/{model_name}/outputs_vllm073[_topp{topp}_seed{seed}]/{eval_dataset}.jsonl
    """
    args = get_args(add_evaluation=True)
    if args.device_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    ## load model
    model_vllm = load_vllm_model(args)
    tokenizer_path = os.path.join(args.tokenizer_folder, args.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    ## load test data
    prompt_list, qid_list = get_prompt_list(args)

    ## run inference
    max_output_len = int(args.max_output_len * args.yarn_factor)
    print("args.max_output_len:", max_output_len)

    if args.topp < 1:
        sampling_params = SamplingParams(temperature=args.temperature, top_p=args.topp, max_tokens=max_output_len,
                                         seed=args.seed)
        print("args.seed:", args.seed)
        print("args.topp:", args.topp)
        print("args.temperature:", args.temperature)

    else:
        sampling_params = SamplingParams(temperature=args.temperature, top_k=args.topk, max_tokens=max_output_len,
                                         seed=args.seed)
        print("Greedy decoding", args.temperature, args.topk)

    output_list = []
    for i in tqdm(range(0, len(prompt_list), args.batch_size)):
        batch_prompts = prompt_list[i:i + args.batch_size]
        if qid_list:
            batch_qids = qid_list[i:i + args.batch_size]

        if args.eval_dataset in ("ifeval", "ifbench", "alpaca_eval", "arena_hard", "mtbench_secondturn", "mtbench_firstturn",
                                 "mmlu", "humaneval", "gsm8k", "mmlu_r1", "aime24", "aime25", "arena_hard_v2",
                                 "lcb5_2408_2502", "lcb6_2502_2505", "ifeval_training", "mmlu_pro", "gpqa_diamond"):
            raw_prompts = batch_prompts
            batch_prompts = [apply_template(prompt, tokenizer, think=args.think) for prompt in batch_prompts]
            for i in range(min(3, len(batch_prompts))):
                print(batch_prompts[i])

        outputs = model_vllm.generate(batch_prompts, sampling_params)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            continue

        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text

            if "<|im_end|>" in generated_text:
                idx = generated_text.index("<|im_end|>")
                generated_text = generated_text[:idx]
            if "<|end_of_text|>" in generated_text:
                idx = generated_text.index("<|end_of_text|>")
                generated_text = generated_text[:idx]
            if "<|eot_id|>" in generated_text:
                idx = generated_text.index("<|eot_id|>")
                generated_text = generated_text[:idx]

            reason = False
            reason_text = ''
            if "</think>" in generated_text:
                idx = generated_text.index("</think>")
                reason_text = generated_text[:idx]
                generated_text = generated_text[idx + len("</think>"):].strip()
                reason = True

            if qid_list:
                qid = batch_qids[j]
                if args.eval_dataset in ("ifeval", "ifeval_training", "ifbench"):
                    output_dict = {"task_id": qid, "prompt": raw_prompts[j], "response": generated_text,
                                   "reason": reason, "reason_text": reason_text}
                elif args.eval_dataset == 'arena_hard':
                    output_dict = {"question_id": qid, "model_id": args.model_name,
                                   "choices": [{"index": 0, "turns": [{"content": generated_text}]}],
                                   "reason": reason, "reason_text": reason_text
                                   }
                elif args.eval_dataset == 'arena_hard_v2':
                    output_dict = {"uid": qid, "model": args.model_name,
                                   "messages": [{"role": "user", "content": raw_prompts[j]},
                                                {"role": "assistant", "content": {"answer": generated_text}}],
                                   "reason": reason, "reason_text": reason_text
                                   }
                elif args.eval_dataset == 'alpaca_eval':
                    output_dict = {"question_id": qid, "model_id": args.model_name,
                                   "instruction": raw_prompts[j], "datasplit": "eval",
                                   "output": generated_text, "reason": reason, "reason_text": reason_text}
                else:
                    output_dict = {"task_id": qid, "output": generated_text,
                                   "reason": reason, "reason_text": reason_text}
                output_list.append(output_dict)
            else:
                output_dict = {"output": generated_text, "reason": reason, "reason_text": reason_text}
                output_list.append(output_dict)

    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    ## write to output_datapath
    if args.topp < 1:
        foldername = "outputs_vllm073_topp{}_seed{}".format(args.topp, args.seed)
    else:
        foldername = "outputs_vllm073"

    if not args.think:
        foldername = "nothink_" + foldername

    output_folder = os.path.join(os.path.join(args.model_folder, args.model_name), foldername)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_name = "%s_%dto%d" % (args.eval_dataset, args.start_idx, args.end_idx) \
        if args.start_idx != -1 and args.end_idx != -1 else args.eval_dataset
    output_name = output_name + ".jsonl"

    output_datapath = os.path.join(output_folder, output_name)

    print("writing to %s" % output_datapath)
    with open(output_datapath, "w", encoding='utf-8') as f:
        for output in output_list:
            if type(output) == dict:
                f.write(json.dumps(output) + "\n")
            else:
                f.write(output + "\n")


if __name__ == "__main__":
    main()
