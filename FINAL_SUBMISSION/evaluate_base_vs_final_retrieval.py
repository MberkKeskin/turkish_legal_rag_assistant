
import sys
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.final_pipeline.final_v8_pipeline import FinalLegalRAGPipelineV8
from app.final_pipeline.benchmark_runner import load_json, normalize_benchmark, get_gold_ids
from app.final_pipeline.metrics import recall_at_k, mrr_score


def get_id(ctx):
    return ctx.get("id") or ctx.get("chunk_id") or ctx.get("doc_id")


def evaluate_system(name, pipeline, data, use_final=True):
    rows = []

    for i, item in enumerate(data, 1):
        q = item["question"]
        gold_ids = get_gold_ids(item)

        if use_final:
            contexts = pipeline.retrieve_contexts(q)
        else:
            # Base mode: use candidates before final selector as a rough base retrieval.
            # If build_v7_candidates is available, we use less processed retrieval.
            from app.final_pipeline.final_v8_pipeline import build_v7_candidates
            candidates = build_v7_candidates(pipeline, q)
            contexts = candidates[:5]

        retrieved_ids = [get_id(c) for c in contexts]

        rows.append({
            "idx": i,
            "System": name,
            "Recall@1": recall_at_k(retrieved_ids, gold_ids, 1),
            "Recall@3": recall_at_k(retrieved_ids, gold_ids, 3),
            "Recall@5": recall_at_k(retrieved_ids, gold_ids, 5),
            "MRR": mrr_score(retrieved_ids, gold_ids),
            "gold_ids": gold_ids,
            "retrieved_ids": retrieved_ids,
            "question": q,
        })

        if i % 50 == 0:
            print(f"{name}: {i}/{len(data)}")

    return pd.DataFrame(rows)


def main():
    result_dir = BASE_DIR / "evaluation_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = BASE_DIR / "data/legal/benchmarks/hard_benchmark_500_v1.json"
    print("Benchmark:", benchmark_path)

    data = normalize_benchmark(load_json(benchmark_path), "hard500")

    pipeline = FinalLegalRAGPipelineV8(
        base_dir=str(BASE_DIR),
        load_generator=False,
        candidate_top_k=100,
        rerank_top_k=5,
        max_expanded_candidates=300,
    )
    pipeline.load()

    base_df = evaluate_system("Base retrieval candidates", pipeline, data, use_final=False)
    final_df = evaluate_system("Final retrieval default_top5", pipeline, data, use_final=True)

    all_df = pd.concat([base_df, final_df], ignore_index=True)

    summary = all_df.groupby("System").agg(
        n=("idx", "count"),
        Recall_1=("Recall@1", "mean"),
        Recall_3=("Recall@3", "mean"),
        Recall_5=("Recall@5", "mean"),
        MRR=("MRR", "mean"),
    ).reset_index()

    all_df.to_csv(result_dir / "base_vs_final_retrieval_details.csv", index=False)
    summary.to_csv(result_dir / "base_vs_final_retrieval_summary.csv", index=False)

    print("\nSUMMARY")
    print(summary)
    print("Saved:", result_dir / "base_vs_final_retrieval_summary.csv")


if __name__ == "__main__":
    main()
