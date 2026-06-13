
import re

from app.final_pipeline.final_rag_pipeline import FinalLegalRAGPipeline, query_type, source_group_from_id
from app.final_pipeline.legal_reranker import LegalEnsembleReranker
from app.bm25_retriever import get_result_id


def dedup_by_id(items):
    seen = set()
    out = []

    for x in items:
        cid = get_result_id(x)
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(x)

    return out


def source_filtered_extra(pipeline, question, source_group, take=80):
    try:
        return pipeline._source_filtered_extra_retrieval(
            question,
            source_group,
            dense_top_k=1000,
            bm25_top_k=1000,
            take=take,
        )
    except Exception:
        return []


def build_v7_candidates(pipeline, question):
    qtype = query_type(question)

    base = pipeline.retriever.retrieve(question, top_k=pipeline.candidate_top_k)
    candidates = pipeline.expand_candidates_source_aware(question, base)

    extra = []

    if qtype == "train_kayit":
        extra += source_filtered_extra(pipeline, question, "train_kayit", take=180)

    elif qtype == "yargitay":
        extra += source_filtered_extra(pipeline, question, "yargitay", take=180)

    elif qtype == "oricon":
        extra += source_filtered_extra(pipeline, question, "oricon", take=120)

    elif qtype == "general":
        extra += source_filtered_extra(pipeline, question, "turkish_law", take=60)
        extra += source_filtered_extra(pipeline, question, "lawchatbot", take=60)

    return dedup_by_id(candidates + extra)[:350]


def select_top3_plus_8_9(reranked50):
    positions = [1, 2, 3, 8, 9]
    selected = []

    for p in positions:
        idx = p - 1
        if 0 <= idx < len(reranked50):
            selected.append(reranked50[idx])

    selected = dedup_by_id(selected)

    if len(selected) < 5:
        selected += reranked50
        selected = dedup_by_id(selected)

    return selected[:5]


def select_default_top5(reranked50):
    selected = dedup_by_id(reranked50)
    return selected[:5]


def extract_article_number(question):
    q = str(question).lower()
    for pat in [r"\bm\.\s*(\d+)", r"\bm\s*(\d+)", r"\bmadde\s*(\d+)"]:
        m = re.search(pat, q)
        if m:
            return m.group(1)
    return None


def sort_by_score(contexts):
    def score(c):
        try:
            return float(c.get("final_rank_score") or c.get("score") or 0.0)
        except Exception:
            return 0.0

    return sorted(contexts, key=score, reverse=True)


def select_contexts_for_llm_v2(question, contexts):
    """
    Same context policy that produced the previous best v8 generation.
    Retrieval returns 5 contexts, then this selector chooses the final LLM context subset.
    """
    qtype = query_type(question)
    contexts = sort_by_score(contexts)

    if qtype == "exact_article":
        article_no = extract_article_number(question)
        exact = []

        for c in contexts:
            cid = str(get_result_id(c)).lower()
            text = str(c.get("text", "")).lower()

            if (
                article_no
                and (
                    f"_m{article_no}" in cid
                    or f"madde {article_no}" in text
                    or f"madde {article_no}-" in text
                    or f"madde {article_no}–" in text
                )
            ):
                exact.append(c)

        if exact:
            return exact[:1], "v2_exact_article_only"

        return contexts[:1], "v2_exact_article_fallback_top1"

    if qtype == "yargitay":
        yargitay_contexts = [
            c for c in contexts
            if source_group_from_id(get_result_id(c)) == "yargitay"
        ]

        if yargitay_contexts:
            return yargitay_contexts[:2], "v2_yargitay_top2"

        return contexts[:2], "v2_yargitay_fallback_top2"

    if qtype == "train_kayit":
        train_contexts = [
            c for c in contexts
            if source_group_from_id(get_result_id(c)) == "train_kayit"
        ]

        if train_contexts:
            return train_contexts[:2], "v2_train_kayit_top2"

        return contexts[:2], "v2_train_kayit_fallback_top2"

    if qtype == "oricon":
        oricon_contexts = [
            c for c in contexts
            if source_group_from_id(get_result_id(c)) == "oricon"
        ]

        if oricon_contexts:
            return oricon_contexts[:2], "v2_oricon_top2"

        return contexts[:2], "v2_oricon_fallback_top2"

    return contexts[:3], "v2_general_top3"


class FinalLegalRAGPipelineV8(FinalLegalRAGPipeline):
    """
    Final v8 pipeline.

    - Source-specific candidate generation v7
    - Error-mined v4 reranker blend
    - Retrieval top5 selector: top3_plus_8_9
    - LLM context selector: selector_v2
    """

    def __init__(
        self,
        base_dir=None,
        load_generator=True,
        candidate_top_k=100,
        rerank_top_k=5,
        enable_source_expansion=True,
        max_expanded_candidates=300,
    ):
        super().__init__(
            base_dir=base_dir,
            load_generator=load_generator,
            candidate_top_k=candidate_top_k,
            rerank_top_k=rerank_top_k,
            enable_source_expansion=enable_source_expansion,
            max_expanded_candidates=max_expanded_candidates,
        )

        self.v8_system_name = (
            "Final v8 - Source-specific Retrieval v7 + "
            "Error-mined v4 Reranker + default_top5 + Selector v2 + Faithful LLM"
        )

    def load(self):
        super().load()

        v4_model_path = self.models_dir / "bge_reranker_legal_ft_v4_error_mined_hf"

        self.reranker = LegalEnsembleReranker(
            v2_reranker_path=str(v4_model_path),
            old_weight=0.85,
            v2_weight=0.15,
            exact_bonus_weight=5.0,
        )

        self.reranker.load()

    def retrieve_contexts(self, question):
        candidates = build_v7_candidates(self, question)
        reranked50 = self.reranker.rerank(question, candidates, top_k=50)
        selected = select_default_top5(reranked50)
        return selected[:5]

    def answer(self, question, context_top_k=5, max_new_tokens=1024):
        retrieved_contexts = self.retrieve_contexts(question)
        llm_contexts, context_policy = select_contexts_for_llm_v2(question, retrieved_contexts)

        answer, raw, context_text, used_retry = self.generator.generate(
            question=question,
            contexts=llm_contexts,
            context_top_k=len(llm_contexts),
            retry=True,
        )

        retrieved_ids = [get_result_id(c) for c in retrieved_contexts]
        llm_context_ids = [get_result_id(c) for c in llm_contexts]

        return {
            "answer": answer,
            "raw": raw,
            "context_text": context_text,
            "used_retry": used_retry,
            "retrieved_contexts": retrieved_contexts,
            "llm_contexts": llm_contexts,
            "retrieved_ids": retrieved_ids,
            "llm_context_ids": llm_context_ids,
            "context_policy": context_policy,
            "system": self.v8_system_name,
        }
