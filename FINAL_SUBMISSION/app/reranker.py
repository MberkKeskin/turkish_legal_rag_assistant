
import re
from typing import List, Dict, Optional

from app.bm25_retriever import get_result_id


LAW_ALIASES = {
    "turk borclar kanunu": [
        "turk borclar kanunu",
        "borclar kanunu",
        "tbk",
        "6098",
    ],
    "ceza muhakemesi kanunu": [
        "ceza muhakemesi kanunu",
        "cmk",
        "5271",
    ],
    "turk medeni kanunu": [
        "turk medeni kanunu",
        "medeni kanun",
        "tmk",
        "4721",
    ],
    "turkiye cumhuriyeti anayasasi": [
        "turkiye cumhuriyeti anayasasi",
        "anayasa",
        "2709",
    ],
    "bilgi edinme hakki kanunu": [
        "bilgi edinme hakki kanunu",
        "4982",
    ],
    "turk bayragi tuzugu": [
        "turk bayragi tuzugu",
        "bayrak tuzugu",
        "85 9034",
    ],
}


def normalize_legal_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace("ı", "i").replace("ğ", "g").replace("ü", "u")
    text = text.replace("ş", "s").replace("ö", "o").replace("ç", "c")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_article_no(question: str) -> Optional[str]:
    q = str(question).lower()

    patterns = [
        r"\bm\.\s*(\d+)",
        r"\bmadde\s*(\d+)",
        r"\bm\s*(\d+)",
    ]

    for pat in patterns:
        match = re.search(pat, q)
        if match:
            return match.group(1)

    return None


def detect_law_key(question: str) -> Optional[str]:
    q = normalize_legal_text(question)

    for law_key, aliases in LAW_ALIASES.items():
        for alias in aliases:
            if normalize_legal_text(alias) in q:
                return law_key

    return None


def item_matches_law(item: Dict, law_key: Optional[str]) -> bool:
    if not law_key:
        return True

    haystack = " ".join([
        str(get_result_id(item) or ""),
        str(item.get("title", "")),
        str(item.get("source", "")),
        str(item.get("text", ""))[:400],
    ])

    h = normalize_legal_text(haystack)
    aliases = LAW_ALIASES.get(law_key, [law_key])

    return any(normalize_legal_text(alias) in h for alias in aliases)


def is_exact_article_and_law_match(question: str, item: Dict) -> bool:
    article_no = extract_article_no(question)
    law_key = detect_law_key(question)

    if not article_no:
        return False

    if not item_matches_law(item, law_key):
        return False

    rid = str(get_result_id(item) or "").lower()
    text = str(item.get("text", "")).lower()
    title = str(item.get("title", "")).lower()
    source = str(item.get("source", "")).lower()

    haystack = " ".join([rid, title, source, text])

    patterns = [
        f"_m{article_no}",
        f"_m{article_no}_",
        f"m.{article_no}",
        f"m {article_no}",
        f"madde {article_no}",
        f"madde {article_no}-",
        f"madde {article_no}–",
        f"madde {article_no}—",
    ]

    return any(p.lower() in haystack for p in patterns)


class CrossEncoderReranker:
    """
    Cross-encoder reranker wrapper.

    Default model:
        BAAI/bge-reranker-v2-m3

    This class is used for System 4 experiments.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        max_length: int = 512,
        trust_remote_code: bool = True,
    ):
        from sentence_transformers import CrossEncoder
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
        )

    def rerank(
        self,
        question: str,
        candidates: List[Dict],
        top_k: int = 5,
        batch_size: int = 8,
    ) -> List[Dict]:
        pairs = []

        for candidate in candidates:
            text = str(candidate.get("text", ""))
            title = str(candidate.get("title", ""))
            source = str(candidate.get("source", ""))

            passage = f"{title}\n{source}\n{text}"
            pairs.append([question, passage])

        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        reranked = []

        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["bm25_score"] = candidate.get("score")
            item["reranker_score"] = float(score)
            item["score"] = float(score)
            item["retriever"] = "crossencoder_reranker"
            reranked.append(item)

        reranked = sorted(
            reranked,
            key=lambda x: x["reranker_score"],
            reverse=True,
        )

        return reranked[:top_k]


def law_aware_protected_rerank(
    question: str,
    candidates: List[Dict],
    reranker: CrossEncoderReranker,
    top_k: int = 5,
    batch_size: int = 8,
) -> List[Dict]:
    """
    Law-aware protected reranking.

    Motivation:
    - Naive semantic reranking can move exact legal article matches down.
    - For legal QA, exact law/article matches must be preserved.
    - Therefore, if BM25 retrieves an exact law+article match, it is protected.
    - Remaining candidates are reranked by the cross-encoder.
    """

    protected = []
    others = []

    for item in candidates:
        if is_exact_article_and_law_match(question, item):
            protected.append(item)
        else:
            others.append(item)

    # Keep only the strongest exact law+article match.
    protected = protected[:1]

    reranked_others = []

    if others:
        reranked_others = reranker.rerank(
            question=question,
            candidates=others,
            top_k=max(top_k * 3, top_k),
            batch_size=batch_size,
        )

    final = []
    seen = set()

    for item in protected + reranked_others:
        rid = get_result_id(item)

        if rid in seen:
            continue

        final.append(item)
        seen.add(rid)

        if len(final) >= top_k:
            break

    return final
