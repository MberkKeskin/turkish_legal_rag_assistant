import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from app.final_pipeline.metrics import (
    bleu_score,
    citation_accuracy,
    exact_match,
    faithfulness_score,
    hallucination_risk,
    mrr_score,
    recall_at_k,
    rouge_scores,
    token_f1,
)


def load_json(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_benchmark(data, name="benchmark"):
    normalized = []

    for i, item in enumerate(data, start=1):
        question_id = (
            item.get("question_id")
            or item.get("soru_id")
            or item.get("query_id")
            or item.get("id")
            or f"{name}_{i:04d}"
        )

        question = item.get("question") or item.get("soru") or item.get("query") or ""
        verified_answer = item.get("verified_answer") or item.get("gold_answer_extract") or item.get("answer") or ""

        gold_sources_raw = item.get("gold_sources")
        gold_chunk_ids = item.get("gold_chunk_ids")

        if isinstance(gold_sources_raw, list) and len(gold_sources_raw) > 0:
            if isinstance(gold_sources_raw[0], dict):
                gold_sources = gold_sources_raw
            else:
                gold_sources = [{"source_id": str(x)} for x in gold_sources_raw]
        elif gold_chunk_ids:
            gold_sources = [{"source_id": str(x)} for x in gold_chunk_ids]
        elif item.get("altin_chunk_id"):
            gold_sources = [{"source_id": item["altin_chunk_id"]}]
        elif item.get("chunk_id"):
            gold_sources = [{"source_id": item["chunk_id"]}]
        else:
            gold_sources = []

        normalized.append(
            {
                "question_id": question_id,
                "question": question,
                "verified_answer": verified_answer,
                "gold_sources": gold_sources,
                "difficulty": item.get("difficulty"),
                "question_type": item.get("question_type"),
                "source": item.get("source"),
                "original_item": item,
            }
        )

    return normalized


def get_gold_ids(item):
    gold_ids = []
    for src in item.get("gold_sources", []):
        gid = src.get("corpus_row_id") or src.get("source_id")
        if gid:
            gold_ids.append(gid)
    return gold_ids


def evaluate_pipeline_on_benchmark(
    pipeline,
    benchmark: List[Dict[str, Any]],
    limit: int = 100,
    system_name: str = "Final Legal RAG Pipeline",
    context_top_k: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []

    for idx, row in enumerate(benchmark[:limit], 1):
        question = row["question"]
        gold_answer = row["verified_answer"]
        gold_ids = get_gold_ids(row)

        result = pipeline.answer(question, context_top_k=context_top_k)

        prediction = result["answer"]
        raw = result["raw"]
        context_text = result.get("context_text", "")
        retrieved_ids = result["retrieved_ids"]

        rouge = rouge_scores(prediction, gold_answer)

        rows.append(
            {
                "idx": idx,
                "question": question,
                "gold_answer": gold_answer,
                "prediction": prediction,
                "raw": raw,
                "gold_ids": gold_ids,
                "retrieved_ids": retrieved_ids,

                "Recall@1": recall_at_k(retrieved_ids, gold_ids, 1),
                "Recall@3": recall_at_k(retrieved_ids, gold_ids, 3),
                "Recall@5": recall_at_k(retrieved_ids, gold_ids, 5),
                "MRR": mrr_score(retrieved_ids, gold_ids),

                "EM": exact_match(prediction, gold_answer),
                "Token_F1": token_f1(prediction, gold_answer),
                "ROUGE1_F": rouge["ROUGE1_F"],
                "ROUGEL_F": rouge["ROUGEL_F"],
                "BLEU": bleu_score(prediction, gold_answer),
                "Faithfulness": faithfulness_score(prediction, context_text),
                "Hallucination_Risk": hallucination_risk(prediction, context_text),
                "Citation_Accuracy": citation_accuracy(retrieved_ids, gold_ids),
                "Used_Retry": int(result.get("used_retry", False)),
            }
        )

        if idx % 10 == 0:
            print(f"{idx}/{limit} done")

    df = pd.DataFrame(rows)

    summary = {
        "System": system_name,
        "n": len(df),

        "Recall@1": df["Recall@1"].mean(),
        "Recall@3": df["Recall@3"].mean(),
        "Recall@5": df["Recall@5"].mean(),
        "MRR": df["MRR"].mean(),

        "EM": df["EM"].mean(),
        "Token_F1": df["Token_F1"].mean(),
        "ROUGE1_F": df["ROUGE1_F"].mean(),
        "ROUGEL_F": df["ROUGEL_F"].mean(),
        "BLEU": df["BLEU"].mean(),
        "Faithfulness": df["Faithfulness"].mean(),
        "Hallucination_Risk": df["Hallucination_Risk"].mean(),
        "Citation_Accuracy": df["Citation_Accuracy"].mean(),
        "Retry_Rate": df["Used_Retry"].mean(),
    }

    summary["Retrieval_R"] = summary["Recall@5"]
    summary["Answer_A_penalized_F1"] = summary["Token_F1"] * summary["Citation_Accuracy"]
    summary["Grounding_G_penalized_Faithfulness"] = summary["Faithfulness"] * summary["Citation_Accuracy"]

    summary["Final_Rubric_Score"] = (
        0.35 * summary["Retrieval_R"]
        + 0.35 * summary["Answer_A_penalized_F1"]
        + 0.30 * summary["Grounding_G_penalized_Faithfulness"]
    )

    return df, summary
