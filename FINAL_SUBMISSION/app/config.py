from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "sample_docs"

# =========================
# Embedding model settings
# =========================
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_MODEL_PATH = str(BASE_DIR / "models" / "bge-m3-legal-ft-system2")
EMBEDDING_LOCAL_ONLY = True
EMBEDDING_TRUST_REMOTE_CODE = False

EMBEDDING_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBEDDING_DOCUMENT_PREFIX = ""
EMBEDDING_NORMALIZE = True

# =========================
# Generation model settings
# =========================
GENERATION_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
GENERATION_MODEL_PATH = ""
GENERATION_LOCAL_ONLY = False

# =========================
# Data mode
# =========================
USE_LEGAL_DATA = True
LEGAL_DATA_PATH = BASE_DIR / "data" / "legal" / "turkish_law_dataset.csv"
LEGAL_HF_DATA_DIR = BASE_DIR / "data" / "legal_hf"
LEGAL_CORPUS_LIMIT = 300
LEGAL_EVAL_LIMIT = 10
LEGAL_TOTAL_EVAL_LIMIT = 12

# =========================
# Retrieval / chunking
# =========================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
USE_LEGAL_CHUNKING = False
LEGAL_SECTION_OVERLAP = 50

TOP_K = 3
MAX_CONTEXT_CHARS = 2000
MAX_CONTEXT_CHUNKS = 6
MAX_CHUNKS_PER_SOURCE = 2
SIMILARITY_THRESHOLD = 0.6

# =========================
# Generation / answer guard
# =========================
MAX_NEW_TOKENS = 128
ENABLE_ANSWER_GUARD = False
ANSWER_MIN_OVERLAP = 1

# =========================
# OOD fallback
# =========================
ENABLE_OOD_FALLBACK = True
OOD_SCORE_THRESHOLD = 0.62
OOD_MIN_CONTEXT_CHARS = 120


# ============================================================
# Strict corpus based System 1 configuration
# ============================================================

FINAL_DATA_DIR = BASE_DIR / "data" / "legal" / "final_datasets"

STRICT_CORPUS_PATH = FINAL_DATA_DIR / "corpus_final_strict_verified.jsonl"
GOLD_BENCHMARK_240_PATH = FINAL_DATA_DIR / "gold_benchmark_final_review_ready_240 (1).json"
RAG_EVAL_1000_PATH = FINAL_DATA_DIR / "rag_eval_final_clean_unique_1000.json"

USE_STRICT_CORPUS = True
