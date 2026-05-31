
# Rubric-Aligned Evaluation Summary

## Evaluation Scenarios

The project was evaluated according to three scenarios:

1. Gold Question + Answer + Relevant Document
   - Retrieval: Recall@k, MRR
   - Answer: Token F1 / ROUGE / BLEU / LLM-based quality
   - Grounding: Faithfulness / citation accuracy
   - Combined score follows the rubric idea: retrieval + answer quality + grounding.

2. Gold Question + Answer
   - Answer quality is evaluated without gold document IDs.
   - Token overlap and semantic/LLM-based quality metrics can be used.

3. No Gold Data
   - Proxy evaluation can be performed using LLM-based relevancy, faithfulness, and coherence.

## Final System Choice

During development, the top3_plus_8_9 selector performed well on Hard100. However, broader Hard500 evaluation showed that default_top5 generalized better.

Hard500:
- default_top5 Recall@5 = 0.760
- top3_plus_8_9 Recall@5 = 0.714

Real30:
- default_top5 Recall@5 = 0.900
- top3_plus_8_9 Recall@5 = 0.900

Therefore, the final system uses default_top5.

## Final Results

| Benchmark | Recall@5 | MRR | Notes |
|---|---:|---:|---|
| Hard100 | 0.850 | 0.703 | Small hard benchmark |
| Hard500 | 0.760 | 0.6126 | Larger and more diverse benchmark |
| Real30 | 0.900 | 0.8733 | Realistic statutory legal questions |

## Interpretation

The final system performs strongly on statutory/practical questions and ORICON-style explanatory sources. The main remaining weakness is retrieval from long and highly similar judicial decision records such as Yargıtay and Constitutional Court / TRAIN-style documents.

## Ablation / Development Summary

The final system was developed through the following stages:

1. Base RAG retrieval and generation
2. Source-aware retrieval
3. Fine-tuned / error-mined reranker
4. Context selection policy
5. Final default_top5 evidence selection

The final selector was chosen based on generalization rather than only optimizing the smaller Hard100 benchmark.
