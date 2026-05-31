# Final Best RAG System

## Selected final configuration

- Retrieval: Source-aware Hybrid Retrieval v3
- Reranker: Ensemble of old BGE reranker and error-mined v4 reranker
- Reranker weights:
  - old reranker weight: 0.85
  - error-mined v4 reranker weight: 0.15
- Candidate top-k: 100
- Expanded candidate max size: 250
- Final rerank top-k: 5
- Context selector: Selector v2
- Generator: Qwen2.5 3B LoRA faithful v2

## Hard100 result

- Recall@1: 0.60
- Recall@3: 0.79
- Recall@5: 0.82
- MRR: 0.689833
- Token F1: 0.560594
- Citation Accuracy: 0.82
- Final Rubric Score: 0.678092

## Development conclusion

The oracle gold-context experiment reached a Final Rubric Score of approximately 0.867, showing that the LLM can produce strong grounded answers when correct evidence is provided. Therefore, the main remaining bottleneck is evidence retrieval and reranking, especially for train_kayit and Yargitay-style cases.