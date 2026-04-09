"""Command-line argument parsing for LLM evaluation."""

import argparse


def get_args(add_evaluation=False):
    """Parse command-line arguments for LLM evaluation.
    
    Args:
        add_evaluation: If True, adds evaluation-specific arguments (default: False)
    
    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - Model configuration (model_folder, model_name, tokenizer_folder, tokenizer_name)
            - Generation parameters (temperature, top-k, top-p, batch-size)
            - Other options (seed, device settings, yarn-factor, think mode)
    """
    parser = argparse.ArgumentParser(description="LLM Evaluation Configuration")

    # Model & tokenizer
    parser.add_argument('--model-folder', type=str, required=True, 
                        help='Directory containing the model')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name of the model subdirectory')
    parser.add_argument('--tokenizer-folder', type=str, required=True,
                        help='Directory containing the tokenizer')
    parser.add_argument('--tokenizer-name', type=str, required=True,
                        help='Name of the tokenizer subdirectory')
    
    # Inference parameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for inference (default: 16)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--use_r1', default=False, action='store_true',
                        help='Use R1-style prompting format')
    parser.add_argument('--device-id', type=str, default=None,
                        help='Comma-separated GPU device IDs (e.g., "0,1,2,3")')
    parser.add_argument('--yarn-factor', type=int, default=1,
                        help='YaRN RoPE scaling factor for extended context (default: 1)')
    parser.add_argument('--no-think',
        dest='think',
        action='store_false',
        default=True,
        help='Disable thinking mode (enabled by default)')
    
    if add_evaluation:
        parser = _add_evaluation_argument(parser)

    args = parser.parse_args()

    return args


def _add_evaluation_argument(parser):
    """Add evaluation-specific command-line arguments.
    
    Args:
        parser: argparse.ArgumentParser instance to add arguments to
    
    Returns:
        argparse.ArgumentParser: Parser with evaluation arguments added
        
    Evaluation arguments include:
        - Benchmark dataset paths (MATH, GSM8K, MMLU, HumanEval, etc.)
        - Inference parameters (temperature, top-k, top-p, max output length)
        - Dataset selection and subsetting options
        - Parallel processing configuration
    """
    group = parser.add_argument_group(title='evaluation')

    # Dataset selection
    group.add_argument('--benchmark-folder', type=str, required=True,
                      help='Root directory containing all benchmark datasets')
    group.add_argument('--eval-dataset', type=str, required=True,
                      help='Name of the evaluation dataset to use')
    
    group.add_argument('--mmlu-path', type=str, default='mmlu/mmlu_test.csv')
    group.add_argument('--mmlupro-path', type=str, default='mmlu_pro/test.json')
    group.add_argument('--mtbench-path', type=str, default='mt_bench/question.jsonl')
    group.add_argument('--arena_hard-path', type=str, default='arena-hard-v0.1/question.jsonl')
    group.add_argument('--arena_hard_v2-path', type=str, default='arena-hard-v2.0/question.jsonl')
    group.add_argument('--aime24-path', type=str, default='aime24/test.jsonl')
    group.add_argument('--aime25-path', type=str, default='aime25/test.jsonl')
    group.add_argument('--gpqa-diamond-path', type=str, default='gpqa/gpqa_diamond.json')
    group.add_argument('--livecodebench-path', type=str, default='livecodebench/test_aug2024tojan2025.json')
    group.add_argument('--livecodebench6-path', type=str, default='livecodebench/test_feb2025toApr2025.json')
    group.add_argument('--ifeval-path', type=str, default='ifeval/input_data.jsonl')
    group.add_argument('--ifbench-path', type=str, default='IFBench/data/IFBench_test.jsonl')
    # Generation parameters
    group.add_argument('--temperature', type=float, default=0,
                      help='Sampling temperature (0 for greedy decoding, default: 0)')
    group.add_argument('--topk', type=int, default=1,
                      help='Top-k sampling parameter (default: 1)')
    group.add_argument('--topp', type=float, default=1,
                      help='Top-p (nucleus) sampling threshold (default: 1)')
    group.add_argument('--max-output-len', type=int, default=2048,
                      help='Maximum output length in tokens (default: 2048)')
    
    # Dataset subsetting
    group.add_argument('--start-idx', type=int, default=-1,
                      help='Starting index for dataset subsetting (default: -1, disabled)')
    group.add_argument('--end-idx', type=int, default=-1,
                      help='Ending index for dataset subsetting (default: -1, disabled)')
    
    # Parallel processing
    group.add_argument('--tensor-parallel-size', type=int, default=1,
                      help='Number of GPUs for tensor parallelism (default: 1)')
    
    # MT-Bench second turn requirement
    group.add_argument('--model-output-path', type=str, default='', nargs='?', const='',
                      help='Path to first turn output (required for mtbench_secondturn)')

    return parser
