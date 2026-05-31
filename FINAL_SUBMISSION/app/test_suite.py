import csv
import json
from pathlib import Path
from typing import Iterable

from .config import (
    LEGAL_DATA_PATH,
    LEGAL_EVAL_LIMIT,
    LEGAL_HF_DATA_DIR,
    LEGAL_TOTAL_EVAL_LIMIT,
    TOP_K,
)
from .data_loader import load_legal_hf_qa_pairs, load_legal_qa_pairs
from .rag_pipeline import RagPipeline


OOD_QUESTIONS = [
    "Türkiye'nin başkenti neresidir?",
    "Dünyanın en yüksek dağı hangisidir?",
    "Python programlama dili hangi yıl çıktı?",
]


def _normalize_source_list(sources: list[str]) -> list[str]:
    normalized: list[str] = []
    for src in sources:
        if " (" in src:
            normalized.append(src.rsplit(" (", 1)[0])
        else:
            normalized.append(src)
    return normalized


def _is_idk(answer: str) -> bool:
    text = answer.strip().lower()
    fallbacks = [
        "bilmiyorum",
        "bu bilgi bağlamda yok",
        "verilen bağlamda bu bilgi bulunmuyor",
        "bağlamda yok",
        "bağlamda bulunmuyor",
    ]
    return any(phrase in text for phrase in fallbacks)


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().replace("\n", " ").split() if t.strip()]


def _f1_score(pred: str, ref: str) -> float:
    pred_tokens = _tokenize(pred)
    ref_tokens = _tokenize(ref)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = {}
    for tok in pred_tokens:
        common[tok] = common.get(tok, 0) + 1
    match = 0
    for tok in ref_tokens:
        if common.get(tok, 0) > 0:
            match += 1
            common[tok] -= 1
    if match == 0:
        return 0.0
    precision = match / len(pred_tokens)
    recall = match / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def _bleu_score(pred: str, ref: str, max_n: int = 4) -> float:
    pred_tokens = _tokenize(pred)
    ref_tokens = _tokenize(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0

    def ngrams(tokens: list[str], n: int) -> list[tuple]:
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    weights = [1 / max_n] * max_n
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        pred_ngrams = ngrams(pred_tokens, n)
        ref_ngrams = ngrams(ref_tokens, n)
        if not pred_ngrams:
            precisions.append(0.0)
            continue
        ref_counts = {}
        for ng in ref_ngrams:
            ref_counts[ng] = ref_counts.get(ng, 0) + 1
        match = 0
        pred_counts = {}
        for ng in pred_ngrams:
            pred_counts[ng] = pred_counts.get(ng, 0) + 1
        for ng, count in pred_counts.items():
            match += min(count, ref_counts.get(ng, 0))
        precisions.append(match / len(pred_ngrams))

    # Brevity penalty
    bp = 1.0
    if len(pred_tokens) < len(ref_tokens):
        bp = pow(2.718281828, 1 - (len(ref_tokens) / max(1, len(pred_tokens))))

    score = bp
    for w, p in zip(weights, precisions):
        if p == 0:
            return 0.0
        score *= p ** w
    return score


def _rouge_n(pred: str, ref: str, n: int) -> float:
    pred_tokens = _tokenize(pred)
    ref_tokens = _tokenize(ref)
    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 0.0
    pred_ngrams = {}
    for i in range(len(pred_tokens) - n + 1):
        key = tuple(pred_tokens[i : i + n])
        pred_ngrams[key] = pred_ngrams.get(key, 0) + 1
    ref_ngrams = {}
    for i in range(len(ref_tokens) - n + 1):
        key = tuple(ref_tokens[i : i + n])
        ref_ngrams[key] = ref_ngrams.get(key, 0) + 1
    overlap = 0
    for k, v in pred_ngrams.items():
        overlap += min(v, ref_ngrams.get(k, 0))
    return overlap / max(1, sum(ref_ngrams.values()))


def _rouge_l(pred: str, ref: str) -> float:
    pred_tokens = _tokenize(pred)
    ref_tokens = _tokenize(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    dp = [[0] * (len(ref_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
    for i in range(1, len(pred_tokens) + 1):
        for j in range(1, len(ref_tokens) + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    return lcs / max(1, len(ref_tokens))


def _groundedness_label(answer: str, context: str) -> str:
    ans_tokens = set(_tokenize(answer))
    ctx_tokens = set(_tokenize(context))
    if not ans_tokens:
        return "unsupported"
    overlap = len(ans_tokens & ctx_tokens)
    if overlap >= 5:
        return "supported"
    if overlap >= 2:
        return "weakly_supported"
    return "unsupported"


def _get_eval_set() -> list[dict]:
    legal_pairs = load_legal_qa_pairs(LEGAL_DATA_PATH, limit=LEGAL_EVAL_LIMIT)
    hf_pairs = load_legal_hf_qa_pairs(LEGAL_HF_DATA_DIR, limit=LEGAL_EVAL_LIMIT)
    eval_set: list[dict] = []
    for item in legal_pairs:
        eval_set.append(
            {
                "question": item["question"],
                "expected_sources": [item["source"]],
                "reference_answer": item.get("answer", ""),
                "in_dataset": True,
            }
        )
    remaining = max(0, LEGAL_TOTAL_EVAL_LIMIT - len(eval_set))
    for item in hf_pairs[:remaining]:
        eval_set.append(
            {
                "question": item["question"],
                "expected_sources": [item["source"]],
                "reference_answer": item.get("answer", ""),
                "in_dataset": True,
            }
        )
    for q in OOD_QUESTIONS:
        eval_set.append(
            {
                "question": q,
                "expected_sources": [],
                "reference_answer": "",
                "in_dataset": False,
            }
        )
    return eval_set


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_retrieval_tests(
    pipeline: RagPipeline, eval_set: list[dict], output_dir: Path
) -> dict:
    results: list[dict] = []
    metrics = {"recall@1": 0, "recall@3": 0, "recall@5": 0, "mrr": 0.0}
    total = 0
    for item in eval_set:
        if not item["in_dataset"]:
            continue
        total += 1
        expected = item["expected_sources"]
        retrieved = pipeline.retrieve(item["question"], top_k=5)
        retrieved_sources = _normalize_source_list(
            [f"{r['source']} ({r['chunk_id']})" for r in retrieved]
        )
        ranks = {}
        for src in expected:
            rank = None
            for i, rsrc in enumerate(retrieved_sources, start=1):
                if rsrc == src:
                    rank = i
                    break
            ranks[src] = rank
        rr = 0.0
        best_rank = min([r for r in ranks.values() if r is not None], default=None)
        if best_rank is not None:
            rr = 1.0 / best_rank
        metrics["mrr"] += rr
        if best_rank is not None and best_rank <= 1:
            metrics["recall@1"] += 1
        if best_rank is not None and best_rank <= 3:
            metrics["recall@3"] += 1
        if best_rank is not None and best_rank <= 5:
            metrics["recall@5"] += 1
        results.append(
            {
                "question": item["question"],
                "expected_sources": expected,
                "retrieved_sources": retrieved_sources,
                "retrieved_ranks": ranks,
                "top_k": 5,
                "recall@1": best_rank is not None and best_rank <= 1,
                "recall@3": best_rank is not None and best_rank <= 3,
                "recall@5": best_rank is not None and best_rank <= 5,
                "reciprocal_rank": rr,
            }
        )

    summary = {}
    if total > 0:
        summary = {
            "recall@1": metrics["recall@1"] / total,
            "recall@3": metrics["recall@3"] / total,
            "recall@5": metrics["recall@5"] / total,
            "hit_rate@1": metrics["recall@1"] / total,
            "hit_rate@3": metrics["recall@3"] / total,
            "hit_rate@5": metrics["recall@5"] / total,
            "mrr": metrics["mrr"] / total,
            "total_questions": total,
        }

    _write_json(output_dir / "retrieval_results.json", {"results": results})
    _write_json(output_dir / "retrieval_summary.json", summary)
    _write_csv(output_dir / "retrieval_results.csv", results)
    return summary


def run_answer_quality_tests(
    pipeline: RagPipeline, eval_set: list[dict], output_dir: Path
) -> dict:
    results: list[dict] = []
    totals = {"em": 0.0, "f1": 0.0, "bleu": 0.0, "r1": 0.0, "r2": 0.0, "rl": 0.0}
    total = 0
    for item in eval_set:
        if not item["in_dataset"] or not item["reference_answer"]:
            continue
        total += 1
        result = pipeline.answer(item["question"], top_k=TOP_K)
        pred = result.answer
        ref = item["reference_answer"]
        em = 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0
        f1 = _f1_score(pred, ref)
        bleu = _bleu_score(pred, ref)
        r1 = _rouge_n(pred, ref, 1)
        r2 = _rouge_n(pred, ref, 2)
        rl = _rouge_l(pred, ref)

        totals["em"] += em
        totals["f1"] += f1
        totals["bleu"] += bleu
        totals["r1"] += r1
        totals["r2"] += r2
        totals["rl"] += rl

        results.append(
            {
                "question": item["question"],
                "reference_answer": ref,
                "generated_answer": pred,
                "exact_match": em,
                "f1": f1,
                "bleu": bleu,
                "rouge_1": r1,
                "rouge_2": r2,
                "rouge_l": rl,
            }
        )

    summary = {}
    if total > 0:
        summary = {
            "exact_match": totals["em"] / total,
            "f1": totals["f1"] / total,
            "bleu": totals["bleu"] / total,
            "rouge_1": totals["r1"] / total,
            "rouge_2": totals["r2"] / total,
            "rouge_l": totals["rl"] / total,
            "total_questions": total,
        }

    _write_json(output_dir / "answer_quality_results.json", {"results": results})
    _write_json(output_dir / "answer_quality_summary.json", summary)
    _write_csv(output_dir / "answer_quality_results.csv", results)
    return summary


def run_rag_reliability_tests(
    pipeline: RagPipeline, eval_set: list[dict], output_dir: Path
) -> dict:
    results: list[dict] = []
    source_match = 0
    ood_ok = 0
    total = 0

    for item in eval_set:
        total += 1
        result = pipeline.answer(item["question"], top_k=TOP_K)
        retrieved_sources = _normalize_source_list(result.sources)
        expected_sources = item["expected_sources"]
        source_match_ok = False
        if item["in_dataset"]:
            source_match_ok = any(src in retrieved_sources for src in expected_sources)
            if source_match_ok:
                source_match += 1
        else:
            if _is_idk(result.answer):
                ood_ok += 1

        retrieved = pipeline.retrieve(item["question"], top_k=TOP_K)
        context = pipeline.build_context(retrieved)
        groundedness = _groundedness_label(result.answer, context)

        results.append(
            {
                "question": item["question"],
                "in_dataset": item["in_dataset"],
                "expected_sources": expected_sources,
                "retrieved_sources": retrieved_sources,
                "final_answer": result.answer,
                "source_match_ok": source_match_ok,
                "ood_fallback_ok": (not item["in_dataset"] and _is_idk(result.answer)),
                "groundedness": groundedness,
            }
        )

    summary = {
        "source_match_accuracy": source_match / max(1, sum(1 for i in eval_set if i["in_dataset"])),
        "ood_fallback_accuracy": ood_ok / max(1, sum(1 for i in eval_set if not i["in_dataset"])),
        "total_questions": total,
    }

    _write_json(output_dir / "rag_reliability_results.json", {"results": results})
    _write_json(output_dir / "rag_reliability_summary.json", summary)
    return summary


def run_all_tests(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = RagPipeline(use_sentence_chunking=True)
    pipeline.build_index()
    eval_set = _get_eval_set()

    retrieval_summary = run_retrieval_tests(pipeline, eval_set, output_dir)
    answer_summary = run_answer_quality_tests(pipeline, eval_set, output_dir)
    rag_summary = run_rag_reliability_tests(pipeline, eval_set, output_dir)

    final_report = {
        "retrieval": retrieval_summary,
        "answer_quality": answer_summary,
        "rag_reliability": rag_summary,
    }
    _write_json(output_dir / "final_test_report.json", final_report)

    md_lines = [
        "# Final Test Report",
        "",
        "## Overview",
        "This report summarizes retrieval, answer quality, and RAG reliability tests.",
        "",
        "## Retrieval Metrics",
        json.dumps(retrieval_summary, indent=2, ensure_ascii=False),
        "",
        "## Answer Quality Metrics",
        json.dumps(answer_summary, indent=2, ensure_ascii=False),
        "",
        "## RAG Reliability Metrics",
        json.dumps(rag_summary, indent=2, ensure_ascii=False),
        "",
        "## Key Observations",
        "- Retrieval metrics are expected to be strong for in-domain questions.",
        "- OOD fallback accuracy reflects how often the model abstains appropriately.",
        "",
        "## Strengths",
        "- Local offline pipeline with reproducible evaluation.",
        "",
        "## Weaknesses",
        "- OOD detection remains challenging in System 1.",
        "",
        "## Recommended Next Steps",
        "- Consider reranking, better embeddings, or calibrated confidence in System 2.",
    ]
    (output_dir / "final_test_report.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[1] / "testResults"
    run_all_tests(results_dir)
