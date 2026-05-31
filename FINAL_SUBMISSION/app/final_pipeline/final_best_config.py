
from pathlib import Path

FINAL_SYSTEM_NAME = "Final Best - Source-aware Retrieval v3 + Error-mined v4 Reranker + Selector v2 + Faithful LLM"

CANDIDATE_TOP_K = 100
RERANK_TOP_K = 5
MAX_EXPANDED_CANDIDATES = 250

# Current best reranker blend
OLD_RERANKER_WEIGHT = 0.85
ERROR_MINED_V4_WEIGHT = 0.15
EXACT_ARTICLE_BONUS_WEIGHT = 5.0

# Model folders under baseline_rag/models
OLD_RERANKER_DIRNAME = "bge-reranker-legal-ft-v1"
ERROR_MINED_V4_RERANKER_DIRNAME = "bge_reranker_legal_ft_v4_error_mined_hf"
FINAL_LLM_LORA_DIRNAME = "qwen2_5_3b_legal_lora_sft_faithful_v2_final"

# Best measured Hard100 metrics
BEST_HARD100_METRICS = {
    "Recall@1": 0.60,
    "Recall@3": 0.79,
    "Recall@5": 0.82,
    "MRR": 0.689833,
    "Token_F1": 0.560594,
    "Citation_Accuracy": 0.82,
    "Final_Rubric_Score": 0.678092,
}
