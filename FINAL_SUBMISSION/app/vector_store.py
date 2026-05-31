import numpy as np
import faiss


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: list[dict] = []

    def add(self, embeddings: np.ndarray, metadatas: list[dict]) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)
        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append(
                {
                    "score": float(score),
                    "chunk_id": meta["chunk_id"],
                    "text": meta["text"],
                    "source": meta["source"],
                }
            )
        return results
