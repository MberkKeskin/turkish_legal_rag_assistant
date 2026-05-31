import re
from pathlib import Path
from typing import Any, Dict, List

from app.bm25_retriever import get_result_id
from app.final_pipeline.hybrid_retriever import HybridLegalRetriever
from app.final_pipeline.legal_reranker import LegalEnsembleReranker
from app.final_pipeline.qwen_generator import QwenLegalGenerator


def extract_article_number_from_question(question: str):
    q = str(question).lower()
    for pat in [r"\bm\.\s*(\d+)", r"\bm\s*(\d+)", r"\bmadde\s*(\d+)"]:
        m = re.search(pat, q)
        if m:
            return m.group(1)
    return None


def source_group_from_id(cid: str) -> str:
    cid = str(cid).lower()

    if cid.startswith("oricon_"):
        return "oricon"
    if cid.startswith("turkish_law_eski"):
        return "turkish_law"
    if cid.startswith("yargitay_"):
        return "yargitay"
    if cid.startswith("train_kayit"):
        return "train_kayit"
    if cid.startswith("lawchatbot_"):
        return "lawchatbot"

    return "other"


def query_type(question: str) -> str:
    q = str(question).lower()

    if extract_article_number_from_question(q):
        return "exact_article"

    if any(x in q for x in [
        "yargıtay", "temyiz", "bozma", "direnme", "hukuk genel kurulu",
        "uyuşmazlığında", "içtihat", "mahkemesi:"
    ]):
        return "yargitay"

    if any(x in q for x in [
        "yaşam hakkı", "mülkiyet hakkı", "adil yargılanma",
        "ifade ve basın", "toplantı ve örgütlenme", "kişi özgürlüğü",
        "kötü muamele", "ihlal", "başvurucu", "temel hak"
    ]):
        return "train_kayit"

    if any(x in q for x in [
        "oricon", "kaynağına göre", "hukuki değerlendirme",
        "ne söylenebilir", "hangi şartlar", "bakımından"
    ]):
        return "oricon"

    return "general"


def yargitay_family_prefix(cid: str):
    """
    Example:
    yargitay_0947_borclar_hukuku_003
    -> yargitay_0947_borclar_hukuku
    """
    cid = str(cid)
    m = re.match(r"^(yargitay_\d+_.+?)_\d+$", cid)
    if m:
        return m.group(1)
    return None


def filter_contexts_for_exact_article(question: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    article_no = extract_article_number_from_question(question)

    if not article_no:
        return contexts

    exact = []

    for c in contexts:
        cid = str(c.get("id") or c.get("chunk_id") or "").lower()
        text = str(c.get("text", "")).lower()

        if (
            f"_m{article_no}" in cid
            or f"madde {article_no}" in text
            or f"madde {article_no}-" in text
            or f"madde {article_no}–" in text
        ):
            exact.append(c)

    return exact[:1] if exact else contexts[:1]


class FinalLegalRAGPipeline:
    """
    End-to-end final RAG pipeline.

    Current final configuration:
    - Hybrid Dense + BM25 retrieval
    - Source-aware candidate expansion
    - Old BGE + V2 legal reranker
    - Source and exact article bonuses
    - Faithful v2 Qwen generator
    """

    def __init__(
        self,
        base_dir: str,
        load_generator: bool = True,
        candidate_top_k: int = 100,
        rerank_top_k: int = 5,
        enable_source_expansion: bool = True,
        max_expanded_candidates: int = 250,
    ):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"

        self.candidate_top_k = candidate_top_k
        self.rerank_top_k = rerank_top_k
        self.enable_source_expansion = enable_source_expansion
        self.max_expanded_candidates = max_expanded_candidates

        self.v2_reranker_path = self.models_dir / "bge_reranker_legal_ft_v4_error_mined_hf"
        self.qwen_lora_path = self.models_dir / "qwen2_5_3b_legal_lora_sft_faithful_v2_final"

        self.retriever = HybridLegalRetriever(
            dense_weight=1.0,
            bm25_weight=1.5,
            dense_k=80,
            bm25_k=80,
            final_k=candidate_top_k,
        )

        self.reranker = LegalEnsembleReranker(
            v2_reranker_path=str(self.v2_reranker_path),
            old_weight=0.85,
            v2_weight=0.15,
            exact_bonus_weight=5.0,
        )

        self.generator = QwenLegalGenerator(
            lora_dir=str(self.qwen_lora_path),
        ) if load_generator else None

    def load(self) -> None:
        self.retriever.build()
        self.reranker.load()

        if self.generator is not None:
            self.generator.load()

    def _deduplicate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []

        for c in candidates:
            rid = get_result_id(c)

            if not rid or rid in seen:
                continue

            seen.add(rid)
            out.append(c)

        return out

    def _find_chunk_by_id(self, target_id: str):
        target_id = str(target_id)

        for c in self.retriever.chunks:
            cid = get_result_id(c)

            if cid == target_id:
                return dict(c)

        return None

    def _all_chunks_by_source_group(self, group_name: str, limit: int = 80) -> List[Dict[str, Any]]:
        out = []

        for c in self.retriever.chunks:
            cid = get_result_id(c)

            if source_group_from_id(cid) == group_name:
                out.append(dict(c))

            if len(out) >= limit:
                break

        return out

    def _source_filtered_extra_retrieval(
        self,
        question: str,
        group_name: str,
        dense_top_k: int = 250,
        bm25_top_k: int = 250,
        take: int = 40,
    ) -> List[Dict[str, Any]]:
        """
        Pull extra candidates from dense and BM25 results, filtered by source group.
        This is used for train_kayit / yargitay / oricon routing.
        """
        extras = []

        try:
            dense_results = self.retriever.dense_retrieve(question, top_k=dense_top_k)
            for c in dense_results:
                cid = get_result_id(c)
                if source_group_from_id(cid) == group_name:
                    x = dict(c)
                    x["expansion_reason"] = f"{group_name}_dense_extra"
                    extras.append(x)
        except Exception:
            pass

        try:
            bm25_results = self.retriever.bm25_retrieve(question, top_k=bm25_top_k, use_expansion=True)
            for c in bm25_results:
                cid = get_result_id(c)
                if source_group_from_id(cid) == group_name:
                    x = dict(c)
                    x["expansion_reason"] = f"{group_name}_bm25_extra"
                    extras.append(x)
        except Exception:
            pass

        return self._deduplicate_candidates(extras)[:take]

    def _yargitay_sibling_expansion(self, candidates: List[Dict[str, Any]], take: int = 50) -> List[Dict[str, Any]]:
        """
        If one part of a Yargitay decision family is retrieved, add sibling parts.
        This targets errors like:
        gold ..._004 but retrieved ..._003
        """
        prefixes = set()

        for c in candidates:
            cid = get_result_id(c)

            if source_group_from_id(cid) != "yargitay":
                continue

            prefix = yargitay_family_prefix(cid)

            if prefix:
                prefixes.add(prefix)

        extras = []

        if not prefixes:
            return extras

        for c in self.retriever.chunks:
            cid = get_result_id(c)

            if not cid:
                continue

            for prefix in prefixes:
                if str(cid).startswith(prefix + "_"):
                    x = dict(c)
                    x["expansion_reason"] = "yargitay_sibling"
                    extras.append(x)
                    break

            if len(extras) >= take:
                break

        return self._deduplicate_candidates(extras)

    def expand_candidates_source_aware(self, question: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        qtype = query_type(question)

        if not self.enable_source_expansion:
            return candidates

        expanded = list(candidates)
        extras = []

        if qtype == "yargitay":
            extras += self._yargitay_sibling_expansion(candidates, take=50)
            extras += self._source_filtered_extra_retrieval(
                question,
                "yargitay",
                dense_top_k=250,
                bm25_top_k=250,
                take=35,
            )

        elif qtype == "train_kayit":
            extras += self._source_filtered_extra_retrieval(
                question,
                "train_kayit",
                dense_top_k=400,
                bm25_top_k=400,
                take=60,
            )

        elif qtype == "oricon":
            extras += self._source_filtered_extra_retrieval(
                question,
                "oricon",
                dense_top_k=250,
                bm25_top_k=250,
                take=40,
            )

        elif qtype == "exact_article":
            # exact article is already handled mainly by BM25 + exact bonus,
            # but keep expansion small to avoid noise.
            pass

        expanded += extras
        expanded = self._deduplicate_candidates(expanded)

        return expanded[:self.max_expanded_candidates]

    def retrieve_contexts(self, question: str) -> List[Dict[str, Any]]:
        candidates = self.retriever.retrieve(question, top_k=self.candidate_top_k)
        candidates = self.expand_candidates_source_aware(question, candidates)
        reranked = self.reranker.rerank(question, candidates, top_k=self.rerank_top_k)
        return reranked

    def serialize_context(self, c: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": get_result_id(c),
            "source": c.get("source", ""),
            "title": c.get("title", ""),
            "text": c.get("text", ""),
            "final_rank_score": c.get("final_rank_score", None),
            "old_score": c.get("old_score", None),
            "v2_score": c.get("v2_score", None),
            "exact_article_bonus": c.get("exact_article_bonus", None),
            "source_compatibility_bonus": c.get("source_compatibility_bonus", None),
            "expansion_reason": c.get("expansion_reason", None),
        }

    def answer(self, question: str, context_top_k: int = 5) -> Dict[str, Any]:
        contexts = self.retrieve_contexts(question)
        serialized_contexts = [self.serialize_context(c) for c in contexts]

        llm_contexts = filter_contexts_for_exact_article(question, serialized_contexts)

        if self.generator is None:
            return {
                "question": question,
                "answer": None,
                "raw": None,
                "contexts": serialized_contexts,
                "llm_contexts": llm_contexts,
                "retrieved_ids": [c["id"] for c in serialized_contexts],
            }

        answer, raw, context_text, used_retry = self.generator.generate(
            question,
            llm_contexts,
            context_top_k=min(context_top_k, len(llm_contexts)),
            retry=True,
        )

        return {
            "question": question,
            "answer": answer,
            "raw": raw,
            "context_text": context_text,
            "used_retry": used_retry,
            "contexts": serialized_contexts,
            "llm_contexts": llm_contexts,
            "retrieved_ids": [c["id"] for c in serialized_contexts],
            "llm_context_ids": [c["id"] for c in llm_contexts],
        }
