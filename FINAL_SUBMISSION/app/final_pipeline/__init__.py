"""
Final RAG pipeline package.

Current final system:
- Hybrid Dense + BM25 retrieval
- Old BGE reranker + V2 legal reranker
- Exact article and source compatibility bonuses
- Faithful Qwen LLM always generates answer
"""

from .hybrid_retriever import HybridLegalRetriever
from .legal_reranker import LegalEnsembleReranker
from .qwen_generator import QwenLegalGenerator
from .final_rag_pipeline import FinalLegalRAGPipeline
