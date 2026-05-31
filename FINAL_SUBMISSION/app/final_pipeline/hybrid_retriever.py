import re
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

from app.rag_pipeline import RagPipeline
from app.bm25_retriever import get_result_id


class HybridLegalRetriever:
    """
    Dense + BM25 hybrid retriever with legal query expansion and RRF fusion.

    This class wraps the existing RagPipeline dense retriever and adds:
    - BM25 over chunk id/source/title/text
    - Turkish legal query expansion
    - Reciprocal Rank Fusion
    """

    def __init__(
        self,
        use_sentence_chunking: bool = True,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.5,
        dense_k: int = 80,
        bm25_k: int = 80,
        final_k: int = 80,
    ):
        self.use_sentence_chunking = use_sentence_chunking
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.final_k = final_k

        self.pipeline_dense = None
        self.bm25 = None
        self.tokenized_corpus = None

    def build(self) -> None:
        self.pipeline_dense = RagPipeline(use_sentence_chunking=self.use_sentence_chunking)
        self.pipeline_dense.build_index()
        self._build_bm25()

    @property
    def chunks(self) -> List[Dict[str, Any]]:
        if self.pipeline_dense is None:
            raise RuntimeError("Retriever is not built. Call build() first.")
        return self.pipeline_dense.chunks

    def _normalize_for_bm25(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"\bm\.\s*(\d+)", r" madde \1 ", text)
        text = re.sub(r"\bm\s*(\d+)", r" madde \1 ", text)
        text = re.sub(r"[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize_bm25(self, text: str) -> List[str]:
        return self._normalize_for_bm25(text).split()

    def _build_bm25(self) -> None:
        bm25_texts = []

        for c in self.chunks:
            full_text = " ".join(
                [
                    str(c.get("chunk_id", "")),
                    str(c.get("id", "")),
                    str(c.get("source", "")),
                    str(c.get("title", "")),
                    str(c.get("text", "")),
                ]
            )
            bm25_texts.append(full_text)

        self.tokenized_corpus = [self._tokenize_bm25(t) for t in bm25_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def expand_legal_query(self, question: str) -> str:
        q = str(question).strip()
        q_low = q.lower()

        expansions = [q]

        article_nums = re.findall(r"\bm\.?\s*(\d+)", q_low)
        article_nums += re.findall(r"\bmadde\s*(\d+)", q_low)

        for n in set(article_nums):
            expansions.append(f"madde {n}")
            expansions.append(f"m {n}")
            expansions.append(f"m.{n}")

        if "yargıtay" in q_low or "temyiz" in q_low or "hukuk genel kurulu" in q_low:
            expansions.append("yargıtay hukuk genel kurulu karar içtihat temyiz bozma direnme")

        if "anayasa" in q_low or "özgürlüğü" in q_low or "hakkı" in q_low:
            expansions.append("anayasa bireysel başvuru temel hak ihlal karar sonucu")

        if "oricon" in q_low or "kaynağına göre" in q_low:
            expansions.append("oricon hukuki değerlendirme şartlar yükümlülükler")

        if "icra" in q_low or "iflas" in q_low or "konkordato" in q_low:
            expansions.append("icra iflas konkordato alacaklı borçlu mahkeme komiser")

        if "idari" in q_low or "iptal davası" in q_low:
            expansions.append("idare hukuku idari işlem dava açma süresi başvuru iptal davası")

        return " ".join(expansions)

    def bm25_retrieve(self, question: str, top_k: int = None, use_expansion: bool = True) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            raise RuntimeError("BM25 is not built. Call build() first.")

        top_k = top_k or self.bm25_k
        query = self.expand_legal_query(question) if use_expansion else question
        query_tokens = self._tokenize_bm25(query)
        scores = self.bm25.get_scores(query_tokens)

        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            c = dict(self.chunks[int(idx)])
            c["score"] = float(scores[int(idx)])
            c["retriever"] = "bm25"
            results.append(c)

        return results

    def dense_retrieve(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
        if self.pipeline_dense is None:
            raise RuntimeError("Dense retriever is not built. Call build() first.")

        top_k = top_k or self.dense_k
        results = self.pipeline_dense.retrieve(question, top_k=top_k)

        out = []
        for r in results:
            c = dict(r)
            c["retriever"] = "dense"
            out.append(c)

        return out

    def rrf_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int = None,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        top_k = top_k or self.final_k

        scores = defaultdict(float)
        items = {}

        for rank, item in enumerate(dense_results, 1):
            rid = get_result_id(item)
            if rid:
                scores[rid] += self.dense_weight * (1.0 / (k + rank))
                items[rid] = item

        for rank, item in enumerate(bm25_results, 1):
            rid = get_result_id(item)
            if rid:
                scores[rid] += self.bm25_weight * (1.0 / (k + rank))
                items[rid] = item

        ranked_ids = sorted(scores.keys(), key=lambda rid: scores[rid], reverse=True)

        fused = []
        for rid in ranked_ids[:top_k]:
            item = dict(items[rid])
            item["score"] = float(scores[rid])
            item["retriever"] = "hybrid_rrf_expanded"
            fused.append(item)

        return fused

    def retrieve(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
        top_k = top_k or self.final_k

        dense_results = self.dense_retrieve(question, top_k=self.dense_k)
        bm25_results = self.bm25_retrieve(question, top_k=self.bm25_k, use_expansion=True)

        return self.rrf_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            top_k=top_k,
        )
