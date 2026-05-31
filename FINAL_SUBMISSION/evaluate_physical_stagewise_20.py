
import sys
import re
import gc
import shutil
import argparse
import subprocess
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = BASE_DIR / "app/final_pipeline/final_v8_pipeline.py"
BACKUP_PATH = BASE_DIR / "app/final_pipeline/final_v8_pipeline.py.physical_benchmark_backup"

STAGES = [
    {
        "stage": "S0_BASE",
        "embedding": "bge-m3",
        "reranker": "bge_reranker_legal_ft_v1_hf",
        "llm": "qwen2_5_3b_legal_lora_sft_500step_final",
        "description": "Base embedding + earliest available reranker + base LLM"
    },
    {
        "stage": "S1_FT_EMBEDDING",
        "embedding": "bge-m3-legal-ft-system2",
        "reranker": "bge_reranker_legal_ft_v1_hf",
        "llm": "qwen2_5_3b_legal_lora_sft_500step_final",
        "description": "Fine-tuned embedding added"
    },
    {
        "stage": "S2_FT_RERANKER",
        "embedding": "bge-m3-legal-ft-system2",
        "reranker": "bge_reranker_legal_ft_v2_retrieval_generated_hf",
        "llm": "qwen2_5_3b_legal_lora_sft_500step_final",
        "description": "Fine-tuned reranker added"
    },
    {
        "stage": "S3_ERROR_MINED_RERANKER",
        "embedding": "bge-m3-legal-ft-system2",
        "reranker": "bge_reranker_legal_ft_v4_error_mined_hf",
        "llm": "qwen2_5_3b_legal_lora_sft_faithful_v2_final",
        "description": "Error-mined reranker and faithful LLM"
    },
    {
        "stage": "S4_FINAL_SYSTEM",
        "embedding": "bge-m3-legal-ft-system2",
        "reranker": "bge_reranker_legal_ft_v4_error_mined_hf",
        "llm": "qwen2_5_3b_legal_lora_sft_faithful_v2_final",
        "description": "Final system with default_top5"
    },
]


def patch_pipeline(stage_cfg):
    text = BACKUP_PATH.read_text(encoding="utf-8")

    replacements = {
        # embedding
        "bge-m3-legal-ft-system2": stage_cfg["embedding"],
        "bge-m3": stage_cfg["embedding"],

        # rerankers
        "bge_reranker_legal_ft_v6_balanced_hf": stage_cfg["reranker"],
        "bge_reranker_legal_ft_v5_rag_eval_error_mined_hf": stage_cfg["reranker"],
        "bge_reranker_legal_ft_v4_error_mined_hf": stage_cfg["reranker"],
        "bge_reranker_legal_ft_v3_error_mined_hf": stage_cfg["reranker"],
        "bge_reranker_legal_ft_v2_retrieval_generated_hf": stage_cfg["reranker"],
        "bge_reranker_legal_ft_v1_hf": stage_cfg["reranker"],

        # LLM LoRA dirs
        "qwen2_5_3b_legal_lora_sft_faithful_v2_final": stage_cfg["llm"],
        "qwen2_5_3b_legal_lora_sft_faithful_v2_final1": stage_cfg["llm"],
        "qwen2_5_3b_legal_lora_sft_500step_final": stage_cfg["llm"],
        "qwen2_5_3b_legal_lora_retrieval_aware_v3_fast_final": stage_cfg["llm"],
    }

    # Important: longer names first to avoid partial replacement issues
    for old in sorted(replacements, key=len, reverse=True):
        text = text.replace(old, replacements[old])

    PIPELINE_PATH.write_text(text, encoding="utf-8")


def restore_pipeline():
    if BACKUP_PATH.exists():
        shutil.copy(BACKUP_PATH, PIPELINE_PATH)


def run_stage(stage, n, benchmark, output_dir):
    cmd = [
        sys.executable,
        "evaluate_one_physical_stage_worker.py",
        "--stage", stage["stage"],
        "--description", stage["description"],
        "--n", str(n),
        "--benchmark", benchmark,
        "--output_dir", output_dir,
    ]

    print("\nRUN:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--benchmark", default="data/legal/benchmarks/hard_benchmark_500_v1.json")
    parser.add_argument("--output_dir", default="evaluation_results/physical_stagewise_hard500_50")
    args = parser.parse_args()

    if not BACKUP_PATH.exists():
        shutil.copy(PIPELINE_PATH, BACKUP_PATH)
        print("Backup created:", BACKUP_PATH)
    else:
        print("Backup exists:", BACKUP_PATH)

    try:
        for stage in STAGES:
            print("\n==============================")
            print("STAGE:", stage["stage"])
            print(stage)
            print("==============================")

            patch_pipeline(stage)

            gc.collect()

            code = run_stage(stage, args.n, args.benchmark, args.output_dir)

            if code != 0:
                print("Stage failed:", stage["stage"], "return code:", code)
                print("Continuing to next stage may be unsafe. Stopping.")
                break

    finally:
        restore_pipeline()
        print("\nPipeline restored from backup.")


if __name__ == "__main__":
    main()
