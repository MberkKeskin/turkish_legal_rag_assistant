# Final Best System v8

## Selected Configuration

- Retrieval: Source-specific Retrieval v7
- Reranker: old BGE reranker + error-mined v4 reranker
- Reranker blend:
  - old reranker weight: 0.85
  - error-mined v4 weight: 0.15
- Candidate generation: source-specific candidate expansion
- Top-5 context selection: top3_plus_8_9
  - uses reranked positions [1, 2, 3, 8, 9]
- Generator: Qwen2.5 3B LoRA faithful v2
- Answering mode: LLM always generates answer from selected contexts

## Hard100 Results

- Recall@1: 0.60
- Recall@3: 0.82
- Recall@5: 0.85
- MRR: 0.703167
- Token F1: 0.565826
- ROUGE1_F: 0.568172
- ROUGEL_F: 0.515485
- BLEU: 0.312124
- Faithfulness: 0.938172
- Hallucination Risk: 0.061828
- Citation Accuracy: 0.85
- Final Rubric Score: 0.705087

## Main Finding

The largest improvement came from retrieval/context selection rather than additional reranker fine-tuning. Diagnostics showed that many missed gold contexts were already inside rerank50 but outside top5. The top3_plus_8_9 strategy improved Recall@5 and Citation Accuracy by pulling useful deeper candidates into the final LLM context.