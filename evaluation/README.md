# LLM Evaluation Framework

This directory contains tools for evaluating large language models on various benchmarks.

## Overview

The evaluation framework supports multiple benchmark datasets across different domains:

- **Math**: AIME24, AIME25 (evaluation scripts provided)
- **Coding**: LiveCodeBench v5, LiveCodeBench v6 (evaluation scripts provided)
- **Multiple Choice**: MMLU, MMLU Pro, GPQA (MMLU evaluation script provided)
- **Instruction Following**: IFEval, IFBench (refer to official evaluation toolkits)
- **General Helpfulness**: Arena-Hard (refer to official evaluation toolkit)

## Installation

Install required dependencies:

```bash
pip install transformers vllm torch tqdm pandas
```

## Directory Structure

```
evaluation/
├── inference.py                    # Main inference script
├── arguments.py                    # Command-line argument definitions
│
├── data/                          # Benchmark datasets and preprocessing
│   ├── benchmark.py               # Dataset preprocessing functions
│   ├── aime24/, aime25/           # AIME competition problems
│   ├── gpqa/                      # GPQA dataset
│   ├── livecodebench/             # LiveCodeBench v5 and v6
│   ├── mmlu/, mmlu_pro/           # MMLU variants
│   ├── arena-hard-v0.1/, arena-hard-v2.0/  # Arena-Hard benchmarks
│   ├── ifeval/, IFBench/          # Instruction following benchmarks
│   └── mt_bench/                  # MT-Bench data
│
├── eval/                          # Evaluation scripts
│   ├── get_scores_math.py         # Math benchmarks (AIME24, AIME25)
│   ├── get_scores_mmlu_batch.py   # MMLU, MMLU-Pro evaluation
│   ├── get_scores_gpqa.py         # GPQA evaluation
│   ├── get_scores_code.py         # Code benchmarks (LiveCodeBench)
│   └── tools/                     # Evaluation utilities
│       ├── grader.py              # Math answer grading
│       ├── code_verifier_utils.py # Code execution and verification
│       └── latex2sympy/           # LaTeX to SymPy conversion
│
├── run.sh                         # Example single benchmark run
├── run_local.sh                   # Local evaluation script
├── run_all.sh                     # Run multiple benchmarks in parallel
└── README.md                      # This file
```

## Usage

### Quick Start

1. Edit `run.sh` to configure your model and data paths
2. Run the evaluation:

```bash
bash run.sh
```

### Advanced Usage

Run inference directly with custom parameters:

```bash
python inference.py \
    --model-folder /path/to/models \
    --model-name your-model \
    --tokenizer-folder /path/to/tokenizers \
    --tokenizer-name your-tokenizer \
    --benchmark-folder /path/to/benchmarks \
    --eval-dataset aime24 \
    --temperature 0.6 \
    --topp 0.95 \
    --batch-size 2048
```

We suggest following the paper config and running benchmarks with k different random seeds.

### Key Arguments

#### Model Configuration (Required)
- `--model-folder`: Directory containing model weights
- `--model-name`: Name of the model subdirectory
- `--tokenizer-folder`: Directory containing tokenizer files
- `--tokenizer-name`: Name of the tokenizer subdirectory

#### Dataset Selection (Required for evaluation)
- `--benchmark-folder`: Root directory containing all benchmark datasets
- `--eval-dataset`: Name of the evaluation dataset (see supported datasets above)

#### Inference Parameters (Optional)
- `--temperature`: Sampling temperature (default: 0 for greedy decoding)
- `--topp`: Top-p (nucleus) sampling threshold (default: 1.0)
- `--topk`: Top-k sampling threshold (default: 1)
- `--max-output-len`: Maximum output length in tokens (default: 2048)
- `--batch-size`: Batch size for inference (default: 16)
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism (default: 1)

#### Dataset Subsetting (Optional)
- `--start-idx`: Starting index for dataset subsetting (default: -1, disabled)
- `--end-idx`: Ending index for dataset subsetting (default: -1, disabled)

#### Other Options
- `--seed`: Random seed for reproducibility (default: 42)
- `--no-think`: Disable thinking mode (flag, thinking enabled by default)
- `--yarn-factor`: Scaling factor for YaRN RoPE extension (default: 1)
- `--device-id`: Comma-separated GPU device IDs (optional)
- `--model-output-path`: Path to first turn output (required for mtbench_secondturn only)

## Supported Datasets

- `aime24` / `aime25`: AIME competition problems
- `lcb5` / `lcb6`: LiveCodeBench (versions 5 and 6)
- `mmlu`: MMLU 5-shot evaluation
- `mmlu_pro`: MMLU Pro dataset
- `gpqa_diamond`: GPQA Diamond subset
- `ifeval`: IFEval instruction following
- `ifbench`: IFBench instruction following
- `arena_hard`: Arena-Hard v0.1

## Running Evaluation Scripts

After generating model outputs using `inference.py`, you can compute metrics using the evaluation scripts in the `eval/` directory.

We also attach our cached generation files in the corresponding model repo for reproducibility. 

### Math Benchmarks (AIME24, AIME25)

Evaluate math problem-solving performance:

```bash
cd eval
python get_scores_math.py \
    --modelfolder /path/to/model/outputs \
    --testfolder /path/to/test_benchmarks
```

This script:
- Evaluates AIME24 and AIME25 benchmarks
- Extracts answers from `\boxed{}` and other formats
- Computes accuracy with mathematical equivalence checking
- Reports mean accuracy and standard deviation across multiple runs

### Multiple Choice (MMLU, MMLU-Pro, GPQA)

Evaluate MMLU and variants:

```bash
cd eval
python get_scores_mmlu_batch.py \
    --modelfolder /path/to/model/outputs \
    --testfolder /path/to/test_benchmarks \
    --verbose  # Optional: print per-category accuracy
```

This script evaluates:
- **MMLU**: Standard MMLU with 4 choices (A-D)
- **MMLU-Pro**: Extended version with up to 16 choices (A-P)

Features:
- Supports boxed answer format (e.g., `\boxed{A}`)
- Extracts letter choices from various formats (parentheses, text, etc.)
- Handles batch-split output files automatically
- Computes accuracy across all MMLU variants
- Optional per-category breakdown with `--verbose` flag

Evaluate GPQA (Graduate-Level Google-Proof Q&A) performance:

```bash
cd eval
python get_scores_gpqa.py \
    --modelfolder /path/to/model/outputs \
    --testfolder /path/to/test_benchmarks
```

This script:
- Evaluates GPQA Diamond subset
- Extracts answers from boxed and text formats
- Uses mathematical equivalence checking for complex answers
- Reports accuracy with standard deviation

### Code Generation (LiveCodeBench)

Evaluate code generation performance:

```bash
cd eval
python get_scores_code.py \
    --modelfolder /path/to/model/outputs \
    --testfolder /path/to/test_benchmarks
```

This script:
- Evaluates LiveCodeBench v5 and v6
- Executes generated code against test cases
- Computes pass rate (percentage of problems solved correctly)
- Reports finish rate (percentage of valid code generations)

**Note**: Code execution requires:
```bash
pip install numpy tqdm
```

### Other Benchmarks

For the following benchmarks, please refer to their official evaluation repositories due to licensing restrictions:

- **Arena-Hard**: Use the [official Arena-Hard evaluation toolkit](https://github.com/lmarena/arena-hard-auto)
- **IFEval**: Use the [official IFEval evaluation script](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
- **IFBench**: Use the [official IFBench evaluation toolkit](https://github.com/instruction-following/IFBench)

These benchmarks require specific evaluation logic and may have licensing terms that restrict redistribution of evaluation code.

## Output Format

Results are saved as JSONL files in:
```
{model_folder}/{model_name}/outputs_vllm073[_topp{topp}_seed{seed}]/{eval_dataset}.jsonl
```

Each line contains:
- `task_id` or `question_id`: Unique identifier for the question
- `output`: Model's generated response
- `reason`: Whether reasoning was used (boolean)
- `reason_text`: The reasoning/thinking content (if applicable)
- Additional dataset-specific fields

## Adding New Datasets

To add a new dataset:

1. Add a preprocessing function in `data/benchmark.py`:
   ```python
   def preprocess_your_dataset(data_file):
       """Preprocess your dataset.
       
       Args:
           data_file: Path to dataset file
       
       Returns:
           tuple: (prompt_list, qid_list) or just prompt_list
       """
       # Your preprocessing logic
       pass
   ```

2. Add the dataset path argument in `arguments.py`:
   ```python
   group.add_argument('--your-dataset-path', type=str, default='path/to/dataset')
   ```

3. Add the dataset case in `inference.py` in the `get_prompt_list()` function:
   ```python
   elif args.eval_dataset == "your_dataset":
       from data.benchmark import preprocess_your_dataset
       input_datapath = os.path.join(args.benchmark_folder, args.your_dataset_path)
       prompt_list, qid_list = preprocess_your_dataset(input_datapath)
   ```

## Notes

- The framework uses vLLM for efficient inference with batching and tensor parallelism support
- Special handling is provided for models like DeepSeek-R1 that require eager mode
- Thinking mode (`<think>` tags) is supported for models trained with reasoning capabilities
- YaRN RoPE scaling is supported for extended context lengths

## License

See the main repository LICENSE file for licensing information.

