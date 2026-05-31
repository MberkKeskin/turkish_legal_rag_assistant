from __future__ import annotations
from app.data_loader import load_strict_corpus

from dataclasses import dataclass
from typing import Iterable
import re

from .chunking import chunk_documents
from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_LOCAL_ONLY,
    EMBEDDING_TRUST_REMOTE_CODE,
    EMBEDDING_QUERY_PREFIX,
    EMBEDDING_DOCUMENT_PREFIX,
    EMBEDDING_NORMALIZE,
    GENERATION_MODEL_NAME,
    GENERATION_MODEL_PATH,
    GENERATION_LOCAL_ONLY,
    LEGAL_CORPUS_LIMIT,
    LEGAL_DATA_PATH,
    USE_LEGAL_CHUNKING,
    LEGAL_SECTION_OVERLAP,
    MAX_CHUNKS_PER_SOURCE,
    MAX_CONTEXT_CHUNKS,
    MAX_CONTEXT_CHARS,
    MAX_NEW_TOKENS,
    ENABLE_ANSWER_GUARD,
    ANSWER_MIN_OVERLAP,
    ENABLE_OOD_FALLBACK,
    OOD_SCORE_THRESHOLD,
    OOD_MIN_CONTEXT_CHARS,
    SIMILARITY_THRESHOLD,
    USE_STRICT_CORPUS,
    STRICT_CORPUS_PATH,
    TOP_K,
    USE_LEGAL_DATA,
)
from .data_loader import load_legal_corpus, load_txt_files
from .embedder import Embedder
from .generator import Generator
from .vector_store import VectorStore


@dataclass
class RagResult:
    answer: str
    sources: list[str]
    retrieved_chunks: list[dict]


class RagPipeline:
    def __init__(self, use_sentence_chunking: bool = True) -> None:
        self.use_sentence_chunking = use_sentence_chunking
        self.embedder = Embedder(
            EMBEDDING_MODEL_NAME,
            model_path=EMBEDDING_MODEL_PATH,
            local_only=EMBEDDING_LOCAL_ONLY,
            trust_remote_code=EMBEDDING_TRUST_REMOTE_CODE,
            query_prefix=EMBEDDING_QUERY_PREFIX,
            document_prefix=EMBEDDING_DOCUMENT_PREFIX,
        )
        # Generator is not loaded during initialization.
        # This keeps retrieval/index building fast and avoids loading LLM on CPU.
        self.generator = None
        self.vector_store: VectorStore | None = None
        self.chunks: list[dict] = []

    @property
    def embedding_device(self) -> str:
        return self.embedder.device

    @property
    def generation_device(self) -> str:
        return self.generator.device

    @property
    def generation_model(self) -> str:
        return self.generator.source

    def build_index(self) -> None:
        if USE_STRICT_CORPUS:
            # System 1 strict corpus mode:
            # uses corpus_final_strict_verified.jsonl directly.
            docs = load_strict_corpus(STRICT_CORPUS_PATH)
            self.chunks = docs
        else:
            if USE_LEGAL_DATA:
                docs = load_legal_corpus(LEGAL_DATA_PATH, limit=LEGAL_CORPUS_LIMIT)
            else:
                docs = load_txt_files(DATA_DIR)

            overlap = LEGAL_SECTION_OVERLAP if USE_LEGAL_CHUNKING else CHUNK_OVERLAP
            self.chunks = chunk_documents(
                docs,
                chunk_size=CHUNK_SIZE,
                overlap=overlap,
                use_sentence_chunking=self.use_sentence_chunking,
                use_legal_chunking=USE_LEGAL_CHUNKING,
            )

        chunk_texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self.embedder.encode_documents(
            chunk_texts,
            normalize=EMBEDDING_NORMALIZE,
        )

        self.vector_store = VectorStore(dim=embeddings.shape[1])
        self.vector_store.add(embeddings, self.chunks)

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        if self.vector_store is None:
            raise RuntimeError("Vector store is not built. Call build_index() first.")

        query_embedding = self.embedder.encode_queries(
            [question],
            normalize=EMBEDDING_NORMALIZE,
        )
        return self.vector_store.search(query_embedding, top_k=top_k)
