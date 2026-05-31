import json
from pathlib import Path
import os
import warnings

VERBOSE = False

from .config import (
    LEGAL_DATA_PATH,
    LEGAL_EVAL_LIMIT,
    LEGAL_HF_DATA_DIR,
    LEGAL_TOTAL_EVAL_LIMIT,
)
from .data_loader import load_legal_hf_qa_pairs, load_legal_qa_pairs
from .rag_pipeline import RagPipeline

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")


OOD_QUESTIONS = [
    "TÃ¼rkiye'nin baÅkenti neresidir?",
    "DÃ¼nyanÄ±n en yÃ¼ksek daÄÄ± hangisidir?",
    "Python programlama dili hangi yÄ±l Ã§Ä±ktÄ±?",
]
def _normalize_source_list(sources: list[str]) -> list[str]:
    normalized: list[str] = []
    for src in sources:
        if " (" in src:
            normalized.append(src.rsplit(" (", 1)[0])
        else:
            normalized.append(src)
    return normalized

def _safe_console(text: str) -> str:
    return text.encode("unicode_escape").decode("ascii")



def _normalize_for_eval(text: str) -> str:
    text = text.strip().lower()
    text = text.strip("\"\'")
    text = " ".join(text.split())
    text = text.rstrip(" .!?;:…")
    return text



def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize_for_eval(pred).split()
    gold_tokens = _normalize_for_eval(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for tok in pred_tokens:
        common[tok] = common.get(tok, 0) + 1
    overlap = 0
    for tok in gold_tokens:
        if common.get(tok, 0) > 0:
            overlap += 1
            common[tok] -= 1
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def _rouge1_f1(pred: str, gold: str) -> float:
    return _token_f1(pred, gold)

def _is_prompt_echo(answer: str) -> bool:
    lowered = answer.lower()
    markers = ("baÄlam:", "soru:", "cevap:", "kullanÄ±cÄ±:", "asistan:")
    return any(m in lowered for m in markers)

def _is_idk(answer: str) -> bool:
    text = _normalize_for_eval(answer)
    fallbacks = [
        "bilmiyorum",
        "bu bilgi baÄlamda yok",
        "verilen baÄlamda bu bilgi bulunmuyor",
        "baÄlamda yok",
        "baÄlamda bulunmuyor",
    ]
    return any(text == phrase or text.startswith(f"{phrase} ") for phrase in fallbacks)




def run_evaluation(top_k: int, results_dir: Path) -> dict:
    pipeline = RagPipeline(use_sentence_chunking=True)
    pipeline.build_index()

    print(f"Embedding device: {pipeline.embedding_device}")
    print(f"Generation device: {pipeline.generation_device}")
    print(f"Generation model: {pipeline.generation_model}")
    print(f"Top_k: {top_k}")
    print()

    legal_pairs = load_legal_qa_pairs(LEGAL_DATA_PATH, limit=LEGAL_EVAL_LIMIT)
    hf_pairs = load_legal_hf_qa_pairs(LEGAL_HF_DATA_DIR, limit=LEGAL_EVAL_LIMIT)
    eval_set: list[dict] = []
    for item in legal_pairs:
        eval_set.append(
            {
                "question": item["question"],
                "answer": item["answer"],
                "expected_sources": [item["source"]],
                "in_dataset": True,
            }
        )
    remaining = max(0, LEGAL_TOTAL_EVAL_LIMIT - len(eval_set))
    for item in hf_pairs[:remaining]:
        eval_set.append(
            {
                "question": item["question"],
                "answer": item["answer"],
                "expected_sources": [item["source"]],
                "in_dataset": True,
            }
        )
    for q in OOD_QUESTIONS:
        eval_set.append({"question": q, "answer": "", "expected_sources": [], "in_dataset": False})

    results: list[dict] = []
    retrieval_matches = 0
    ood_correct = 0
    exact_matches = 0
    f1_scores: list[float] = []
    rouge1_scores: list[float] = []
    prompt_echoes = 0

    for item in eval_set:
        result = pipeline.answer(item["question"], top_k=top_k)
        retrieved_sources = _normalize_source_list(result.sources)
        expected_sources = item["expected_sources"]
        in_dataset = item["in_dataset"]
        expected_answer = item.get("answer", "")

        retrieval_ok = True
        if in_dataset:
            retrieval_ok = any(src in retrieved_sources for src in expected_sources)
            if retrieval_ok:
                retrieval_matches += 1
            if expected_answer:
                pred_norm = _normalize_for_eval(result.answer)
                gold_norm = _normalize_for_eval(expected_answer)
                if pred_norm == gold_norm:
                    exact_matches += 1
                f1_scores.append(_token_f1(result.answer, expected_answer))
                rouge1_scores.append(_rouge1_f1(result.answer, expected_answer))
        else:
            if _is_idk(result.answer):
                ood_correct += 1
        if _is_prompt_echo(result.answer):
            prompt_echoes += 1

        results.append(
            {
                "question": item["question"],
                "top_k": top_k,
                "expected_sources": expected_sources,
                "in_dataset": in_dataset,
                "retrieved_sources": retrieved_sources,
                "answer": result.answer,
                "retrieval_ok": retrieval_ok,
                "idk_ok": (not in_dataset and _is_idk(result.answer)),
            }
        )

        if VERBOSE:
            print("Question:", _safe_console(item["question"]))
            print("Expected sources:", _safe_console(", ".join(expected_sources)) if expected_sources else "(none)")
            print("Retrieved sources:", _safe_console(", ".join(retrieved_sources)))
            print("Answer:", _safe_console(result.answer))
        if VERBOSE:
            if in_dataset:
                print("Retrieval OK:", retrieval_ok)
            else:
                print('Said "I don\'t know":', _is_idk(result.answer))
            print("-" * 60)

    summary = {
        "top_k": top_k,
        "total_questions": len(eval_set),
        "in_dataset_questions": sum(1 for item in eval_set if item["in_dataset"]),
        "out_of_dataset_questions": sum(1 for item in eval_set if not item["in_dataset"]),
        "retrieval_matches": retrieval_matches,
        "ood_idk_correct": ood_correct,
        "exact_match": exact_matches,
        "token_f1_avg": (sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0,
        "rouge1_avg": (sum(rouge1_scores) / len(rouge1_scores)) if rouge1_scores else 0.0,
        "prompt_echo_count": prompt_echoes,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"eval_results_topk_{top_k}.json"
    output_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print("Summary:")
    print(json.dumps(summary, indent=2))
    print(f"Saved results to: {output_path}")
    print()

    return {"summary": summary, "results_path": str(output_path)}


if __name__ == "__main__":
    results_root = Path(__file__).resolve().parents[1] / "eval_results"
    run_evaluation(top_k=3, results_dir=results_root)
    run_evaluation(top_k=5, results_dir=results_root)
