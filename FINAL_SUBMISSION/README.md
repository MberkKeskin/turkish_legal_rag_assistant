# Turkish Legal RAG Assistant

This project is a Turkish legal Retrieval-Augmented Generation system. It answers Turkish legal questions by retrieving relevant legal sources, selecting evidence passages, and generating grounded answers from those sources.

The system also supports uploaded document question answering. A user can upload a TXT, PDF, or DOCX file and ask questions over that document.

---

## 1. Main Features

### Legal Knowledge Base RAG

The user asks a legal question. The system:

1. Searches the legal knowledge base.
2. Retrieves relevant legal source passages.
3. Reranks and selects the most useful evidence.
4. Generates a grounded legal answer.
5. Shows the answer and the sources used.

### Uploaded Document RAG

The user uploads a TXT, PDF, or DOCX document. The system:

1. Extracts text from the uploaded document.
2. Splits the document into chunks.
3. Retrieves the chunks most relevant to the user question.
4. Generates an answer based only on the uploaded document.
5. Shows the answer and the document sections used.

---

## 2. Final System Configuration

The final system uses:

- Source-aware legal retrieval
- Error-mined reranker ensemble
- default_top5 evidence selection
- Selector v2 for final LLM context selection
- Faithful legal answer generation

During development, the top3_plus_8_9 selector performed well on the smaller Hard100 benchmark. However, the broader Hard500 benchmark showed that default_top5 generalized better. Therefore, the final system uses default_top5.

---

## 3. Key Evaluation Results

| Benchmark | Recall@5 | MRR | Notes |
|---|---:|---:|---|
| Hard100 | 0.850 | 0.703 | Small hard benchmark |
| Hard500 | 0.760 | 0.6126 | Larger and more diverse benchmark |
| Real30 | 0.900 | 0.8733 | Realistic statutory legal questions |

The results show that the system performs strongly on statutory and practical legal questions. The main remaining challenge is retrieval from long and highly similar judicial decision records such as Yargıtay and Constitutional Court / TRAIN-style documents.

---

## 4. Project Structure

```text
FINAL_SUBMISSION/
  README.md
  requirements.txt
  final_ui.py
  smoke_test.py
  smoke_test_light.py
  app/
  data/
  models/
  testResults/

---

## Physical Checkpoint-Based Stagewise Evaluation

To analyze the contribution of each major model component, a physical checkpoint-based stagewise benchmark was performed on a stratified 50-question subset from Hard500.

The compared stages are:

| Stage | Description |
|---|---|
| S0_BASE | Base embedding + earliest available reranker + base LLM |
| S1_FT_EMBEDDING | Fine-tuned embedding added |
| S2_FT_RERANKER | Fine-tuned reranker added |
| S3_ERROR_MINED_RERANKER | Error-mined reranker and faithful LLM |
| S4_FINAL_SYSTEM | Final system with default_top5 |

Results are available under:

```text
evaluation_results/physical_stagewise_stratified50/physical_stagewise_summary.csv
evaluation_results/physical_stagewise_stratified50/physical_stagewise_details.csv
```

This benchmark reports:

- Recall@1
- Recall@3
- Recall@5
- MRR
- Token F1
- Citation accuracy proxy
- Faithfulness proxy
- Final rubric proxy

For the same-LLM comparison requirement, S0, S1 and S2 should be compared because they keep the generation component fixed while changing retrieval, embedding, and reranking components. S3 and S4 represent later full-system improvements.

---

## Custom Benchmark Evaluation

A custom benchmark file can be evaluated with:

```bash
python evaluate_custom_benchmark.py --benchmark my_benchmark.json --output my_results.csv
```

If GPU memory is limited or only retrieval is required:

```bash
python evaluate_custom_benchmark.py --benchmark my_benchmark.json --output my_results.csv --no_generator
```

Expected benchmark JSON format:

```json
[
  {
    "question": "Türk Borçlar Kanunu m.314 kapsamında ifa zamanı nasıl düzenlenmiştir?",
    "gold_answer": "Kiracı, aksine sözleşme ve yerel adet olmadıkça kira bedelini her ayın sonunda ödemekle yükümlüdür.",
    "gold_ids": ["turkish_law_eski_6098_turk_borclar_kanunu_m314"]
  }
]
```

If gold_ids are provided, Recall@k and MRR are calculated. If gold_answer is provided, Token F1 is calculated.

---

## Colab Demo Notebook

A clean final demo notebook is provided under:

```text
notebooks/final_demo_colab.ipynb
```

It includes:

1. Mount Google Drive
2. Install requirements
3. Run light smoke test
4. Run full smoke test
5. Launch the manual UI
