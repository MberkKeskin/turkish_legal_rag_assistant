
import sys
import re
import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.final_pipeline.final_v8_pipeline import FinalLegalRAGPipelineV8
from app.final_pipeline.benchmark_runner import load_json, normalize_benchmark, get_gold_ids
from app.final_pipeline.metrics import recall_at_k, mrr_score


def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-ZğüşöçıİĞÜŞÖÇ0-9\s]", " ", text)
    return [t for t in text.split() if t.strip()]


def token_f1(pred, gold):
    pred_toks = tokenize(pred)
    gold_toks = tokenize(gold)

    if not pred_toks or not gold_toks:
        return 0.0

    counts = {}
    for t in pred_toks:
        counts[t] = counts.get(t, 0) + 1

    overlap = 0
    for t in gold_toks:
        if counts.get(t, 0) > 0:
            overlap += 1
            counts[t] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def get_gold_answer(item):
    # Hard benchmark fields
    for k in [
        "verified_answer",
        "gold_answer_extract",
        "gold_answer",
        "answer",
        "expected_answer",
        "cevap",
        "gold"
    ]:
        if k in item and item[k]:
            return str(item[k])
    return ""


def citation_proxy(retrieved_ids, gold_ids):
    if not gold_ids:
        return None
    return 1.0 if any(g in retrieved_ids for g in gold_ids) else 0.0


def rubric_proxy(recall5, token_f1_value, grounding):
    r = 0.0 if recall5 is None else float(recall5)
    a = 0.0 if token_f1_value is None else float(token_f1_value)
    g = 0.0 if grounding is None else float(grounding)

    # Rubric idea: A and G are penalized by retrieval.
    a_penalized = a * r
    g_penalized = g * r

    return 0.35 * r + 0.4 * a_penalized + 0.25 * g_penalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--benchmark", default="data/legal/benchmarks/hard_benchmark_500_v1.json")
    parser.add_argument("--output_dir", default="evaluation_results/physical_stagewise_hard500_50")
    args = parser.parse_args()

    out_dir = BASE_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    details_path = out_dir / "physical_stagewise_details.csv"
    summary_path = out_dir / "physical_stagewise_summary.csv"

    benchmark_path = BASE_DIR / args.benchmark
    data = normalize_benchmark(load_json(benchmark_path), benchmark_path.stem)
    data = data[:args.n]

    existing = pd.DataFrame()
    rows = []
    done = set()

    if details_path.exists():
        existing = pd.read_csv(details_path)
        rows = existing.to_dict("records")

        stage_existing = existing[existing["Stage"] == args.stage]
        for _, r in stage_existing.iterrows():
            done.add(int(r["idx"]))

        print("Existing rows:", len(existing))
        print("Done for stage:", len(done))

    print("Benchmark:", benchmark_path)
    print("Stage:", args.stage)
    print("N:", len(data))
    print("Loading pipeline...")

    pipeline = FinalLegalRAGPipelineV8(
        base_dir=str(BASE_DIR),
        load_generator=True,
        candidate_top_k=100,
        rerank_top_k=5,
        max_expanded_candidates=300,
    )
    pipeline.load()

    for idx, item in enumerate(data, 1):
        if idx in done:
            print("skip", args.stage, idx)
            continue

        q = item["question"]
        gold_ids = get_gold_ids(item)
        gold_answer = get_gold_answer(item)

        print(f"{args.stage}: question {idx}/{len(data)}")

        try:
            out = pipeline.answer(q)
            answer = out.get("answer", "")
            retrieved_ids = out.get("retrieved_ids", [])
            llm_context_ids = out.get("llm_context_ids", [])
            used_retry = out.get("used_retry", False)
            error = None
        except Exception as e:
            answer = ""
            retrieved_ids = []
            llm_context_ids = []
            used_retry = False
            error = str(e)

        r1 = recall_at_k(retrieved_ids, gold_ids, 1)
        r3 = recall_at_k(retrieved_ids, gold_ids, 3)
        r5 = recall_at_k(retrieved_ids, gold_ids, 5)
        mrr = mrr_score(retrieved_ids, gold_ids)
        f1 = token_f1(answer, gold_answer) if gold_answer else None
        grounding = citation_proxy(retrieved_ids, gold_ids)
        final_score = rubric_proxy(r5, f1, grounding)

        rows.append({
            "Stage": args.stage,
            "Description": args.description,
            "idx": idx,
            "Question_ID": item.get("question_id"),
            "Question": q,
            "Gold_Answer": gold_answer,
            "Answer": answer,
            "Gold_IDs": gold_ids,
            "Retrieved_IDs": retrieved_ids,
            "LLM_Context_IDs": llm_context_ids,
            "Recall@1": r1,
            "Recall@3": r3,
            "Recall@5": r5,
            "MRR": mrr,
            "Token_F1": f1,
            "Citation_Accuracy_Proxy": grounding,
            "Faithfulness_Proxy": grounding,
            "Final_Rubric_Proxy": final_score,
            "Used_Retry": used_retry,
            "Error": error,
        })

        pd.DataFrame(rows).to_csv(details_path, index=False)

    df = pd.DataFrame(rows)

    summary = df.groupby(["Stage", "Description"]).agg(
        n=("idx", "count"),
        Recall_1=("Recall@1", "mean"),
        Recall_3=("Recall@3", "mean"),
        Recall_5=("Recall@5", "mean"),
        MRR=("MRR", "mean"),
        Token_F1=("Token_F1", "mean"),
        Citation_Accuracy_Proxy=("Citation_Accuracy_Proxy", "mean"),
        Faithfulness_Proxy=("Faithfulness_Proxy", "mean"),
        Final_Rubric_Proxy=("Final_Rubric_Proxy", "mean"),
    ).reset_index()

    summary.to_csv(summary_path, index=False)

    print("\nSUMMARY")
    print(summary)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
