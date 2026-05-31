
from app.bm25_retriever import BM25Retriever, get_result_id


def rrf_fusion(result_lists, rrf_k=60, top_k=20):
    """
    Standard Reciprocal Rank Fusion.
    """
    fused = {}
    objects = {}

    for results in result_lists:
        for rank, item in enumerate(results, 1):
            rid = get_result_id(item)

            if rid is None:
                continue

            if rid not in fused:
                fused[rid] = 0.0
                objects[rid] = dict(item)

            fused[rid] += 1.0 / (rrf_k + rank)

    ranked_ids = sorted(fused.keys(), key=lambda rid: fused[rid], reverse=True)

    final_results = []
    for rid in ranked_ids[:top_k]:
        item = objects[rid]
        item["score"] = fused[rid]
        item["retriever"] = "hybrid_rrf"
        final_results.append(item)

    return final_results


def weighted_rrf_fusion(result_lists_with_weights, rrf_k=60, top_k=20):
    """
    Weighted Reciprocal Rank Fusion.
    Useful when BM25 should dominate exact legal citation queries.
    """
    fused = {}
    objects = {}

    for results, weight in result_lists_with_weights:
        for rank, item in enumerate(results, 1):
            rid = get_result_id(item)

            if rid is None:
                continue

            if rid not in fused:
                fused[rid] = 0.0
                objects[rid] = dict(item)

            fused[rid] += weight * (1.0 / (rrf_k + rank))

    ranked_ids = sorted(fused.keys(), key=lambda rid: fused[rid], reverse=True)

    final_results = []
    for rid in ranked_ids[:top_k]:
        item = objects[rid]
        item["score"] = fused[rid]
        item["retriever"] = "weighted_hybrid_rrf"
        final_results.append(item)

    return final_results


class HybridRetriever:
    """
    Combines dense FAISS retrieval from RagPipeline with BM25 lexical retrieval.
    """

    def __init__(self, pipeline, bm25_retriever=None):
        self.pipeline = pipeline
        self.bm25_retriever = bm25_retriever or BM25Retriever(pipeline.chunks)

    def dense_retrieve(self, query, top_k=20):
        results = self.pipeline.retrieve(query, top_k=top_k)
        for result in results:
            result["retriever"] = "dense"
        return results

    def bm25_retrieve(self, query, top_k=20):
        return self.bm25_retriever.retrieve(query, top_k=top_k)

    def retrieve_rrf(self, query, dense_k=30, bm25_k=30, final_k=20, rrf_k=60):
        dense_results = self.dense_retrieve(query, top_k=dense_k)
        bm25_results = self.bm25_retrieve(query, top_k=bm25_k)

        return rrf_fusion(
            [dense_results, bm25_results],
            rrf_k=rrf_k,
            top_k=final_k
        )

    def retrieve_weighted_rrf(
        self,
        query,
        dense_k=30,
        bm25_k=30,
        final_k=20,
        rrf_k=60,
        dense_weight=0.5,
        bm25_weight=2.0
    ):
        dense_results = self.dense_retrieve(query, top_k=dense_k)
        bm25_results = self.bm25_retrieve(query, top_k=bm25_k)

        return weighted_rrf_fusion(
            [
                (dense_results, dense_weight),
                (bm25_results, bm25_weight),
            ],
            rrf_k=rrf_k,
            top_k=final_k
        )
