import re
from typing import Any, Dict, List

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from app.bm25_retriever import get_result_id


LAW_HINTS = {
    "türk borçlar kanunu": "borclar",
    "borçlar kanunu": "borclar",
    "türk medeni kanunu": "medeni",
    "medeni kanunu": "medeni",
    "ceza muhakemesi kanunu": "ceza_muhakemesi",
    "cmk": "ceza_muhakemesi",
    "anayasa": "anayasa",
    "bilgi edinme hakkı kanunu": "bilgi_edinme",
    "türk bayrağı tüzüğü": "bayragi",
}


def minmax_norm(arr):
    arr = np.array(arr, dtype=float)
    if arr.max() - arr.min() < 1e-9:
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def extract_article_number(question: str):
    q = str(question).lower()
    for pat in [r"\bm\.\s*(\d+)", r"\bm\s*(\d+)", r"\bmadde\s*(\d+)"]:
        m = re.search(pat, q)
        if m:
            return m.group(1)
    return None


def detect_law_hint(question: str):
    q = str(question).lower()
    for key, val in LAW_HINTS.items():
        if key in q:
            return val
    return None


class LegalEnsembleReranker:
    """
    Current final reranker:
    - Old BGE reranker
    - V2 legal fine-tuned reranker
    - Exact article bonus
    - Source compatibility bonus
    """

    def __init__(
        self,
        v2_reranker_path: str,
        old_model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 512,
        old_weight: float = 0.85,
        v2_weight: float = 0.15,
        exact_bonus_weight: float = 5.0,
        device: str = None,
    ):
        self.v2_reranker_path = v2_reranker_path
        self.old_model_name = old_model_name
        self.max_length = max_length
        self.old_weight = old_weight
        self.v2_weight = v2_weight
        self.exact_bonus_weight = exact_bonus_weight
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.old_reranker = None
        self.v2_reranker = None

    def load(self) -> None:
        self.old_reranker = CrossEncoder(
            self.old_model_name,
            max_length=self.max_length,
            device=self.device,
        )

        self.v2_reranker = CrossEncoder(
            str(self.v2_reranker_path),
            max_length=self.max_length,
            device=self.device,
        )

    def exact_article_bonus(self, question: str, item: Dict[str, Any]) -> float:
        article_no = extract_article_number(question)
        law_hint = detect_law_hint(question)

        if not article_no:
            return 0.0

        rid = str(get_result_id(item)).lower()
        source = str(item.get("source", "")).lower()
        title = str(item.get("title", "")).lower()
        text = str(item.get("text", "")).lower()

        article_match = (
            f"_m{article_no}" in rid
            or f"madde {article_no}" in text
            or f"madde {article_no}-" in text
            or f"madde {article_no}–" in text
        )

        if not article_match:
            return 0.0

        if law_hint:
            joined = " ".join([rid, source, title])
            if law_hint not in joined:
                return 0.0

        return self.exact_bonus_weight

    def source_compatibility_bonus(self, question: str, item: Dict[str, Any]) -> float:
        q = str(question).lower()

        rid = str(get_result_id(item)).lower()
        source = str(item.get("source", "")).lower()
        title = str(item.get("title", "")).lower()

        joined = " ".join([rid, source, title])

        bonus = 0.0

        if any(x in q for x in ["yargıtay", "temyiz", "bozma", "direnme", "hukuk genel kurulu", "uyuşmazlığında"]):
            if rid.startswith("yargitay_") or "yargıtay" in joined or "hukuk genel kurulu" in joined:
                bonus += 0.25

        if any(x in q for x in ["yaşam hakkı", "mülkiyet hakkı", "adil yargılanma", "ifade", "toplantı", "özgürlüğü", "ihlal"]):
            if rid.startswith("train_kayit"):
                bonus += 0.35

        if any(x in q for x in ["hangi şartlar", "hangi yükümlülük", "ne söylenebilir", "hukuki değerlendirme", "bakımından"]):
            if rid.startswith("oricon_"):
                bonus += 0.20

        if extract_article_number(q):
            if "turkish_law_eski" in rid:
                bonus += 0.25

        domain_terms = [
            "icra", "iflas", "medeni", "borclar", "ticaret",
            "idare", "ceza", "anayasa", "kamulastirma"
        ]

        for term in domain_terms:
            if term in q and term in joined:
                bonus += 0.05

        return bonus

    def rerank(self, question: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if self.old_reranker is None or self.v2_reranker is None:
            raise RuntimeError("Rerankers are not loaded. Call load() first.")

        old_pairs = []
        v2_pairs = []

        for c in candidates:
            text = str(c.get("text", ""))
            source = str(c.get("source", ""))
            title = str(c.get("title", ""))
            passage = f"{source}\n{title}\n{text}"

            old_pairs.append([question, passage])
            v2_pairs.append([question, passage])

        old_scores = np.array(
            self.old_reranker.predict(old_pairs, batch_size=16, show_progress_bar=False)
        ).reshape(-1)

        v2_scores = np.array(
            self.v2_reranker.predict(v2_pairs, batch_size=16, show_progress_bar=False)
        ).reshape(-1)

        old_norm = minmax_norm(old_scores)
        v2_norm = minmax_norm(v2_scores)

        reranked = []

        for c, oscore, vscore, on, vn in zip(candidates, old_scores, v2_scores, old_norm, v2_norm):
            item = dict(c)

            exact_bonus = self.exact_article_bonus(question, item)
            source_bonus = self.source_compatibility_bonus(question, item)

            item["old_score"] = float(oscore)
            item["v2_score"] = float(vscore)
            item["old_norm"] = float(on)
            item["v2_norm"] = float(vn)
            item["exact_article_bonus"] = float(exact_bonus)
            item["source_compatibility_bonus"] = float(source_bonus)

            item["final_rank_score"] = (
                self.old_weight * float(on)
                + self.v2_weight * float(vn)
                + float(exact_bonus)
                + float(source_bonus)
            )

            reranked.append(item)

        return sorted(reranked, key=lambda x: x["final_rank_score"], reverse=True)[:top_k]
