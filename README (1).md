# Turkish Legal RAG Assistant

This repository contains the final submission package for the Turkish Legal RAG project.

The main project files are located under:

```text
FINAL_SUBMISSION/
```

---

## Repository Structure

```text
FINAL_SUBMISSION/
  README.md
  requirements.txt
  final_ui.py
  smoke_test.py
  smoke_test_light.py
  evaluate_custom_benchmark.py
  evaluate_physical_stagewise_20.py
  evaluate_one_physical_stage_worker.py
  app/
  data/
  evaluation_results/
  testResults/
  models/   # not included in GitHub due to file size

notebooks/
  final_demo_colab.ipynb
```

---

## Important: Model Files

Large model files are not included in this GitHub repository due to file size limitations.

Download or copy the `models/` folder from the following Google Drive link:

https://drive.google.com/drive/folders/1ptbpRlGl3L6f9-Ei2fm4czbZeDhez2oF?usp=drive_link

After downloading, place the folder exactly as:

```text
FINAL_SUBMISSION/models/
```

The system expects this path. Without the `models/` folder, the full pipeline cannot be executed.

---

## Quick Start

Clone the repository:

```bash
git clone https://github.com/MberkKeskin/turkish_legal_rag_assistant.git
cd turkish_legal_rag_assistant/FINAL_SUBMISSION
```

Install requirements:

```bash
pip install -r requirements.txt
```

Run the lightweight smoke test:

```bash
python smoke_test_light.py
```

Run the full smoke test if GPU memory is sufficient:

```bash
python smoke_test.py
```

Run the manual Colab / notebook UI:

```python
exec(open('final_ui.py', encoding='utf-8').read())
```

---

## Google Colab Demo

A clean demo notebook is provided at:

```text
notebooks/final_demo_colab.ipynb
```

The notebook includes:

1. Installing requirements
2. Running the light smoke test
3. Running the full smoke test
4. Launching the manual UI

If running on Colab, make sure the `models/` folder is available under:

```text
FINAL_SUBMISSION/models/
```

---

## Main Features

- Turkish legal Retrieval-Augmented Generation system
- Legal knowledge-base question answering
- Uploaded document question answering for TXT, PDF, and DOCX files
- Source-aware retrieval
- Fine-tuned embedding and reranker components
- Error-mined reranker stage
- Faithful answer generation
- Custom benchmark evaluation script
- Physical checkpoint-based stagewise benchmark results

---

## Evaluation Results

Final evaluation files are available under:

```text
FINAL_SUBMISSION/evaluation_results/
```

The physical checkpoint-based stagewise benchmark is available at:

```text
FINAL_SUBMISSION/evaluation_results/physical_stagewise_stratified50/physical_stagewise_summary.csv
FINAL_SUBMISSION/evaluation_results/physical_stagewise_stratified50/physical_stagewise_details.csv
```

The compared stages are:

| Stage | Description |
|---|---|
| S0_BASE | Base embedding + earliest available reranker + base LLM |
| S1_FT_EMBEDDING | Fine-tuned embedding added |
| S2_FT_RERANKER | Fine-tuned reranker added |
| S3_ERROR_MINED_RERANKER | Error-mined reranker and faithful LLM |
| S4_FINAL_SYSTEM | Final system with default_top5 |

---

## Custom Benchmark Evaluation

A custom benchmark file can be evaluated with:

```bash
python evaluate_custom_benchmark.py --benchmark my_benchmark.json --output my_results.csv
```

For retrieval-only evaluation:

```bash
python evaluate_custom_benchmark.py --benchmark my_benchmark.json --output my_results.csv --no_generator
```

Expected JSON format:

```json
[
  {
    "question": "Türk Borçlar Kanunu m.314 kapsamında ifa zamanı nasıl düzenlenmiştir?",
    "gold_answer": "Kiracı, aksine sözleşme ve yerel adet olmadıkça kira bedelini her ayın sonunda ödemekle yükümlüdür.",
    "gold_ids": ["turkish_law_eski_6098_turk_borclar_kanunu_m314"]
  }
]
```

---

## Notes

The full system requires GPU memory because it loads embedding, reranker, and LLM components.

If GPU memory is limited, use:

```bash
python smoke_test_light.py
```

The full smoke test may use around 12 GB GPU memory.
