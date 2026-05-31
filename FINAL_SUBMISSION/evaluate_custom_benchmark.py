
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import re

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.final_pipeline.final_v8_pipeline import FinalLegalRAGPipelineV8


def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-ZğüşöçıİĞÜŞÖÇ0-9\s]", " ", text)
    return [t for t in text.split() if t.strip()]


def token_f1(pred, gold):
    pred_toks = tokenize(pred)
    gold_toks = tokenize(gold)

    if not pred_toks or not gold_toks:
        return 0.0

    common = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1

    overlap = 0
    for t in gold_toks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)

    return 2 * precision * recall / (precision + recall)


def recall_at_k(retrieved, gold_ids, k):
    if not gold_ids:
        return None
    topk = retrieved[:k]
    return 1.0 if any(g in topk for g in gold_ids) else 0.0


def mrr(retrieved, gold_ids):
    if not gold_ids:
        return None
    for i, rid in enumerate(retrieved, 1):
        if rid in gold_ids:
            return 1.0 / i
    return 0.0


def load_benchmark(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSON file")
    parser.add_argument("--output", default="custom_benchmark_results.csv")
    parser.add_argument("--no_generator", action="store_true", help="Only run retrieval, skip answer generation")
    args = parser.parse_args()

    data = load_benchmark(args.benchmark)

    pipeline = FinalLegalRAGPipelineV8(
        base_dir=str(BASE_DIR),
        load_generator=not args.no_generator,
        candidate_top_k=100,
        rerank_top_k=5,
        max_expanded_candidates=300,
    )
    pipeline.load()

    rows = []

    for i, item in enumerate(data, 1):
        question = item.get("question") or item.get("soru")
        gold_answer = item.get("gold_answer") or item.get("answer") or item.get("altin_cevap")
        gold_ids = item.get("gold_ids") or item.get("gold_doc_ids") or item.get("altin_chunk_id") or []

        if isinstance(gold_ids, str):
            gold_ids = [gold_ids]

        if not question:
            continue

        if args.no_generator:
            contexts = pipeline.retrieve_contexts(question)
            retrieved_ids = [c.get("id") or c.get("chunk_id") for c in contexts]
            answer = ""
        else:
            out = pipeline.answer(question)
            retrieved_ids = out.get("retrieved_ids", [])
            answer = out.get("answer", "")

        row = {
            "idx": i,
            "question": question,
            "answer": answer,
            "gold_answer": gold_answer,
            "retrieved_ids": retrieved_ids,
            "gold_ids": gold_ids,
            "Recall@1": recall_at_k(retrieved_ids, gold_ids, 1),
            "Recall@3": recall_at_k(retrieved_ids, gold_ids, 3),
            "Recall@5": recall_at_k(retrieved_ids, gold_ids, 5),
            "MRR": mrr(retrieved_ids, gold_ids),
            "Token_F1": token_f1(answer, gold_answer) if gold_answer and answer else None,
        }

        rows.append(row)

        print(f"{i}/{len(data)} done")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    print("\nSaved:", args.output)

    metric_cols = ["Recall@1", "Recall@3", "Recall@5", "MRR", "Token_F1"]
    print("\nSummary:")
    print(df[metric_cols].mean(numeric_only=True))


if __name__ == "__main__":
    main()
