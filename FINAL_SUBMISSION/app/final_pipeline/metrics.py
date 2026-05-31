import re
from typing import Dict, List

import sacrebleu
from rouge_score import rouge_scorer


def normalize_text_for_metric(text: str) -> str:
    text = str(text).lower()

    text = re.sub(r"\n\s*kaynak\s*:.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[?\s*id\s*:\s*[^\]\n]+\]?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[?\s*kaynak\s*:\s*[^\]\n]+\]?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[?\s*başlık\s*:\s*[^\]\n]+\]?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"seçilen kaynak\s*:[^\n]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"cevap\s*:", " ", text, flags=re.IGNORECASE)

    text = re.sub(r"[^\wçğıöşüâîû]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_text_for_metric(pred).split()
    gold_tokens = normalize_text_for_metric(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    pred_counts = {}
    gold_counts = {}

    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1

    common = 0
    for t in pred_counts:
        common += min(pred_counts.get(t, 0), gold_counts.get(t, 0))

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_text_for_metric(pred) == normalize_text_for_metric(gold))


def rouge_scores(pred: str, gold: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    s = scorer.score(str(gold), str(pred))
    return {
        "ROUGE1_F": s["rouge1"].fmeasure,
        "ROUGEL_F": s["rougeL"].fmeasure,
    }


def bleu_score(pred: str, gold: str) -> float:
    try:
        return sacrebleu.sentence_bleu(str(pred), [str(gold)]).score / 100.0
    except Exception:
        return 0.0


def faithfulness_score(answer: str, context_text: str) -> float:
    ans_tokens = normalize_text_for_metric(answer).split()
    ctx_tokens = normalize_text_for_metric(context_text).split()

    if len(ans_tokens) == 0:
        return 0.0

    ctx_set = set(ctx_tokens)
    meaningful = [t for t in ans_tokens if len(t) >= 2]

    if len(meaningful) == 0:
        meaningful = ans_tokens

    overlap = sum(1 for t in meaningful if t in ctx_set)

    return overlap / len(meaningful)


def hallucination_risk(answer: str, context_text: str) -> float:
    return 1.0 - faithfulness_score(answer, context_text)


def recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> int:
    return int(any(rid in gold_ids for rid in retrieved_ids[:k]))


def mrr_score(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in gold_ids:
            return 1.0 / i
    return 0.0


def citation_accuracy(retrieved_ids: List[str], gold_ids: List[str]) -> int:
    return int(any(rid in gold_ids for rid in retrieved_ids))
