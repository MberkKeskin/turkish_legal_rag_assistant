from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        local_only: bool = False,
        trust_remote_code: bool = False,
        device: str | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name
        self.model_path = model_path or ""
        self.query_prefix = query_prefix or ""
        self.document_prefix = document_prefix or ""

        if self.model_path and Path(self.model_path).exists():
            source = self.model_path
            local_only = True
        else:
            source = model_name

        self.source = source
        self.model = SentenceTransformer(
            source,
            device=device,
            cache_folder=None,
            trust_remote_code=trust_remote_code,
            local_files_only=local_only,
        )

    def _prepare_queries(self, texts: list[str]) -> list[str]:
        if not self.query_prefix:
            return texts
        return [self.query_prefix + text for text in texts]

    def _prepare_documents(self, texts: list[str]) -> list[str]:
        if not self.document_prefix:
            return texts
        return [self.document_prefix + text for text in texts]

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        )
        return embeddings.astype("float32")

    def encode_queries(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        prepared = self._prepare_queries(texts)
        return self.encode(prepared, normalize=normalize)

    def encode_documents(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        prepared = self._prepare_documents(texts)
        return self.encode(prepared, normalize=normalize)
