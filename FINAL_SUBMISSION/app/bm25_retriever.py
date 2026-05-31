
import re
import numpy as np
from rank_bm25 import BM25Okapi


def get_result_id(item):
    return item.get("id") or item.get("chunk_id")


def normalize_for_bm25(text):
    """
    Normalizes Turkish legal text for BM25 retrieval.
    Keeps Turkish characters and strengthens article-number patterns such as m.314 / Madde 314.
    """
    text = str(text).lower()

    # m.314 -> m 314 madde 314
    text = re.sub(r"\bm\.\s*(\d+)", r"m \1 madde \1", text)

    # madde 314 -> madde 314 m 314
    text = re.sub(r"\bmadde\s+(\d+)", r"madde \1 m \1", text)

    # Keep letters, numbers and Turkish chars
    text = re.sub(r"[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_bm25(text):
    return normalize_for_bm25(text).split()


class BM25Retriever:
    """
    BM25 retriever over already prepared RAG chunks.
    It uses chunk_id, id, source, title and text fields together for better exact legal citation matching.
    """

    def __init__(self, chunks):
        self.chunks = chunks
        self.bm25_texts = []

        for chunk in chunks:
            full_text = " ".join([
                str(chunk.get("chunk_id", "")),
                str(chunk.get("id", "")),
                str(chunk.get("source", "")),
                str(chunk.get("title", "")),
                str(chunk.get("text", "")),
            ])
            self.bm25_texts.append(full_text)

        self.tokenized_corpus = [tokenize_bm25(text) for text in self.bm25_texts]
        self.index = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, top_k=20):
        query_tokens = tokenize_bm25(query)
        scores = self.index.get_scores(query_tokens)

        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            chunk = dict(self.chunks[int(idx)])
            chunk["score"] = float(scores[int(idx)])
            chunk["retriever"] = "bm25"
            results.append(chunk)

        return results
