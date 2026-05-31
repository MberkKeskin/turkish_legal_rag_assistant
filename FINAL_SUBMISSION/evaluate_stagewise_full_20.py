
import sys
import re
import json
import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.final_pipeline.final_v8_pipeline import FinalLegalRAGPipelineV8, build_v7_candidates
from app.final_pipeline.benchmark_runner import load_json, normalize_benchmark, get_gold_ids
from app.final_pipeline.metrics import recall_at_k, mrr_score
from app.bm25_retriever import get_result_id


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
    for k in ["gold_answer", "answer", "expected_answer", "cevap", "gold"]:
        if k in item and item[k]:
            return str(item[k])
    return ""


def dedup_by_id(contexts):
    seen = set()
    out = []

    for c in contexts:
        cid = get_result_id(c)
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(c)

    return out


def select_positions(contexts, positions):
    selected = []

    for p in positions:
        idx = p - 1
        if 0 <= idx < len(contexts):
            selected.append(contexts[idx])

    selected = dedup_by_id(selected)

    if len(selected) < 5:
        selected += contexts
        selected = dedup_by_id(selected)

    return selected[:5]


def stage_contexts(pipeline, question):
    candidates = build_v7_candidates(pipeline, question)
    candidates = dedup_by_id(candidates)

    reranked50 = pipeline.reranker.rerank(question, candidates, top_k=50)
    reranked50 = dedup_by_id(reranked50)

    stages = {
        "S0_raw_candidates_top5": candidates[:5],
        "S1_reranker_top5": reranked50[:5],
        "S2_top3_plus_8_9_selector": select_positions(reranked50, [1, 2, 3, 8, 9]),
        "S3_final_default_top5": reranked50[:5],
    }

    return stages


def safe_generate(pipeline, question, contexts):
    try:
        answer, raw, context_text, used_retry = pipeline.generator.generate(
            question=question,
            contexts=contexts,
            context_top_k=len(contexts),
            retry=True,
        )
        return answer, used_retry, None
    except Exception as e:
        return "", False, str(e)


def citation_accuracy_proxy(retrieved_ids, gold_ids):
    if not gold_ids:
        return None
    return 1.0 if any(g in retrieved_ids for g in gold_ids) else 0.0


def rubric_score(recall5, token_f1_value, grounding_value):
    # Rubric idea: Final = 0.35R + 0.4A + 0.25G
    # A and G are penalized by retrieval.
    r = 0.0 if recall5 is None else float(recall5)
    a = 0.0 if token_f1_value is None else float(token_f1_value)
    g = 0.0 if grounding_value is None else float(grounding_value)

    a_penalized = a * r
    g_penalized = g * r

    return 0.35 * r + 0.4 * a_penalized + 0.25 * g_penalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/legal/benchmarks/hard_benchmark_100_v1.json")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--output_dir", default="evaluation_results/stagewise_full_20")
    args = parser.parse_args()

    benchmark_path = BASE_DIR / args.benchmark
    output_dir = BASE_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    details_path = output_dir / "stagewise_full_20_details.csv"
    summary_path = output_dir / "stagewise_full_20_summary.csv"

    data = normalize_benchmark(load_json(benchmark_path), benchmark_path.stem)
    data = data[:args.n]

    print("Benchmark:", benchmark_path)
    print("N:", len(data))

    existing = pd.DataFrame()

    if details_path.exists():
        existing = pd.read_csv(details_path)
        print("Existing rows:", len(existing))

    done_pairs = set()

    if len(existing):
        for _, r in existing.iterrows():
            done_pairs.add((int(r["idx"]), str(r["Stage"])))

    print("Loading final pipeline with generator...")

    pipeline = FinalLegalRAGPipelineV8(
        base_dir=str(BASE_DIR),
        load_generator=True,
        candidate_top_k=100,
        rerank_top_k=5,
        max_expanded_candidates=300,
    )
    pipeline.load()

    rows = []

    if len(existing):
        rows = existing.to_dict("records")

    for idx, item in enumerate(data, 1):
        question = item["question"]
        gold_ids = get_gold_ids(item)
        gold_answer = get_gold_answer(item)

        print(f"\nQuestion {idx}/{len(data)}")

        stages = stage_contexts(pipeline, question)

        for stage_name, contexts in stages.items():
            if (idx, stage_name) in done_pairs:
                print("skip", idx, stage_name)
                continue

            retrieved_ids = [get_result_id(c) for c in contexts]

            print("running", stage_name)

            answer, used_retry, error = safe_generate(pipeline, question, contexts)

            r1 = recall_at_k(retrieved_ids, gold_ids, 1)
            r3 = recall_at_k(retrieved_ids, gold_ids, 3)
            r5 = recall_at_k(retrieved_ids, gold_ids, 5)
            mrr = mrr_score(retrieved_ids, gold_ids)

            f1 = token_f1(answer, gold_answer) if gold_answer else None
            grounding = citation_accuracy_proxy(retrieved_ids, gold_ids)

            final_score = rubric_score(r5, f1, grounding)

            rows.append({
                "idx": idx,
                "Stage": stage_name,
                "Question": question,
                "Gold_Answer": gold_answer,
                "Answer": answer,
                "Gold_IDs": gold_ids,
                "Retrieved_IDs": retrieved_ids,
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

    summary = df.groupby("Stage").agg(
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
    print("Saved details:", details_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
