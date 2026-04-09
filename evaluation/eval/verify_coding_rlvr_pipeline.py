#!/usr/bin/env python3
"""Sanity checks for coding RLVR pipeline recovery."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import pickle
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CONVERTER = REPO_ROOT / "evaluation" / "data" / "coding" / "ensure_competitive_coding_verl_parquet.py"
LIVEBENCH_CONVERTER = REPO_ROOT / "evaluation" / "data" / "livebench" / "convert_livebench_to_verl_parquet.py"
REWARD_MODULE_PATH = REPO_ROOT / "evaluation" / "eval" / "verl_custom_reward.py"


def _encode_ground_truth(inputs: list[str], outputs: list[str], fn_name: str = "") -> str:
    payload = {"inputs": inputs, "outputs": outputs, "fn_name": fn_name}
    blob = pickle.dumps(json.dumps(payload, ensure_ascii=False))
    return base64.b64encode(zlib.compress(blob)).decode("utf-8")


def _load_reward_module():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("_eval_verl_custom_reward", str(REWARD_MODULE_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load reward module: {REWARD_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def _assert_verl_schema(path: Path) -> int:
    table = pq.read_table(path)
    cols = set(table.column_names)
    required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    missing = required - cols
    if missing:
        raise AssertionError(f"Missing required columns in {path}: {sorted(missing)}")
    return table.num_rows


def run_synthetic_checks() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="coding_rlvr_verify_") as td:
        tdp = Path(td)

        # 1) train converter check (non-VERL input -> converted manifest)
        src_dir = tdp / "src_train"
        src_dir.mkdir(parents=True, exist_ok=True)
        src_parquet = src_dir / "train-00000-of-00001.parquet"

        source_rows = [
            {
                "question": "Read one line and print it.",
                "test_input": ["hello\\n", "world\\n"],
                "test_output": ["hello\\n", "world\\n"],
                "test_method": "stdio",
                "test_time_limit": 4,
                "id": "synthetic-0",
            }
        ]
        pq.write_table(pa.Table.from_pylist(source_rows), src_parquet)

        out_dir = tdp / "converted_train"
        manifest = out_dir / "manifest.json"
        code, stdout, stderr = _run(
            [
                sys.executable,
                str(TRAIN_CONVERTER),
                "--input_dir",
                str(src_dir),
                "--output_dir",
                str(out_dir),
                "--manifest_path",
                str(manifest),
                "--data_source",
                "nemotron_cascade_rl_coding",
                "--split",
                "train",
            ]
        )
        if code != 0:
            raise RuntimeError(f"train converter failed\nstdout={stdout}\nstderr={stderr}")

        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        train_files = [Path(p) for p in manifest_data.get("train_files", [])]
        if not train_files:
            raise AssertionError("train converter manifest has empty train_files")
        train_rows = _assert_verl_schema(train_files[0])

        # 2) livebench converter check
        livebench_json = tdp / "livebench.json"
        livebench_json.write_text(
            json.dumps(
                [
                    {
                        "question": "Read a line and print it exactly.",
                        "test_input": ["abc\\n", "x y\\n"],
                        "test_output": ["abc\\n", "x y\\n"],
                        "test_time_limit": 4,
                        "test_method": "stdio",
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        livebench_parquet = tdp / "livebench_verl.parquet"
        code, stdout, stderr = _run(
            [
                sys.executable,
                str(LIVEBENCH_CONVERTER),
                "--input_json",
                str(livebench_json),
                "--output_parquet",
                str(livebench_parquet),
                "--data_source",
                "livebench/code_generation_lite",
            ]
        )
        if code != 0:
            raise RuntimeError(f"livebench converter failed\nstdout={stdout}\nstderr={stderr}")
        livebench_rows = _assert_verl_schema(livebench_parquet)

        # 3) reward sanity check
        reward = _load_reward_module()
        gt = _encode_ground_truth(["hello\\n"], ["hello\\n"], "")
        ok_solution = "```python\nprint(input())\n```"
        bad_solution = "```python\nprint('wrong')\n```"

        ok_train = reward.compute_score(
            data_source="nemotron_cascade_rl_coding",
            solution_str=ok_solution,
            ground_truth=gt,
            extra_info={"prompt_language": "en", "test_time_limit": 4},
        )
        bad_train = reward.compute_score(
            data_source="nemotron_cascade_rl_coding",
            solution_str=bad_solution,
            ground_truth=gt,
            extra_info={"prompt_language": "en", "test_time_limit": 4},
        )
        ok_val = reward.compute_score(
            data_source="livebench/code_generation_lite",
            solution_str=ok_solution,
            ground_truth=gt,
            extra_info={"prompt_language": "en", "test_time_limit": 4},
        )

        if float(ok_train.get("answer_reward", 0.0)) < 1.0:
            raise AssertionError(f"expected correct train answer_reward=1.0, got {ok_train}")
        if float(bad_train.get("score", 1.0)) != 0.0:
            raise AssertionError(f"expected incorrect train score=0.0, got {bad_train}")
        if float(ok_val.get("score", 0.0)) < 1.0:
            raise AssertionError(f"expected correct val score=1.0, got {ok_val}")

        return {
            "train_manifest_mode": manifest_data.get("mode"),
            "train_rows": train_rows,
            "livebench_rows": livebench_rows,
            "reward_ok_train": ok_train,
            "reward_bad_train": bad_train,
            "reward_ok_val": ok_val,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify coding RLVR pipeline recovery")
    parser.add_argument(
        "--check_live_data",
        action="store_true",
        help="Also run converter against live /131_data path (requires real parquet payload)",
    )
    parser.add_argument(
        "--train_parquet_dir",
        type=str,
        default="/131_data/geeho/data/Nemotron-RL-coding-competitive_coding/data",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    summary = {
        "synthetic": run_synthetic_checks(),
    }

    if args.check_live_data:
        manifest_path = REPO_ROOT / "evaluation" / "data" / "coding" / "_live_manifest.json"
        code, stdout, stderr = _run(
            [
                sys.executable,
                str(TRAIN_CONVERTER),
                "--input_dir",
                args.train_parquet_dir,
                "--manifest_path",
                str(manifest_path),
            ]
        )
        summary["live_data_check"] = {
            "exit_code": code,
            "stdout": stdout[-2000:],
            "stderr": stderr[-2000:],
            "manifest_path": str(manifest_path),
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
