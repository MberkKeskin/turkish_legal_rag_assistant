# Turkish Legal RAG Assistant

A final submission package for a Turkish Legal Retrieval-Augmented Generation system.

This project provides a source-grounded legal question-answering assistant for Turkish legal texts. The system supports both:

1. **Legal knowledge-base question answering**
2. **Uploaded document question answering** for TXT, PDF, and DOCX files

The repository also includes instructor-facing Colab notebooks for remote testing and benchmark evaluation.

---

## Project Summary

The system follows a Retrieval-Augmented Generation pipeline:

```text
User Question
      ↓
Fine-tuned Legal Embedding Retriever
      ↓
Hybrid / Source-aware Retrieval
      ↓
Fine-tuned Legal Reranker
      ↓
Context Selection
      ↓
Fine-tuned Faithful LLM Generator
      ↓
Grounded Legal Answer + Sources
```

The goal is not only to generate an answer, but to generate an answer that is grounded in retrieved legal sources.

---

## Main Features

* Turkish legal RAG pipeline
* Legal knowledge-base QA
* Uploaded document QA
* TXT, PDF, and DOCX document support
* Fine-tuned embedding model
* Fine-tuned reranker
* Error-mined reranker improvement stage
* Fine-tuned LoRA legal answer generator
* Source-aware retrieval and context selection
* Lightweight smoke test
* Full smoke test
* Custom benchmark script
* Instructor benchmark UI
* Click-to-run Google Colab notebooks

---

## Repository Structure

```text
turkish_legal_rag_assistant/
│
├── README.md
│
├── FINAL_SUBMISSION/
│   ├── README.md
│   ├── requirements.txt
│   ├── final_ui.py
│   ├── smoke_test.py
│   ├── smoke_test_light.py
│   ├── evaluate_custom_benchmark.py
│   ├── evaluate_physical_stagewise_20.py
│   ├── evaluate_one_physical_stage_worker.py
│   │
│   ├── app/
│   │   ├── final_pipeline/
│   │   └── ...
│   │
│   ├── data/
│   ├── evaluation_results/
│   ├── testResults/
│   │
│   └── models/
│       └── model files are downloaded automatically from Hugging Face in Colab
│
└── notebooks/
    ├── qa_demo_click_run.ipynb
    └── benchmark_demo_click_run.ipynb
```

---

## Model Files

Large model files are not stored directly in GitHub because of repository file size limits.

The required models are hosted on Hugging Face:

| Component                  | Hugging Face Repository                                |
| -------------------------- | ------------------------------------------------------ |
| Fine-tuned embedding model | `Berk2003/bge-m3-legal-ft-system2`                     |
| Fine-tuned reranker        | `Berk2003/bge-reranker-legal-ft-v4-error-mined-hf`     |
| Fine-tuned LoRA generator  | `Berk2003/qwen2-5-3b-legal-lora-sft-faithful-v2-final` |

The Colab notebooks automatically download these model repositories into:

```text
FINAL_SUBMISSION/models/
```

Expected local model structure:

```text
FINAL_SUBMISSION/models/
  bge-m3-legal-ft-system2/
  bge_reranker_legal_ft_v4_error_mined_hf/
  qwen2_5_3b_legal_lora_sft_faithful_v2_final/
```

---

## Recommended Way to Run

The recommended way to run the project is through the provided Google Colab notebooks.

### 1. Question Answer Demo Notebook

```text
notebooks/Demo QuestionAnswerRAG.ipynb
```

This notebook:

1. Clones the GitHub repository
2. Installs requirements
3. Downloads the models from Hugging Face
4. Verifies model files
5. Runs the lightweight smoke test
6. Launches the final Question Answer UI

Recommended demo question:

```text
Türk Borçlar Kanunu m.314 kapsamında ifa zamanı nasıl düzenlenmiştir?
```

Uploaded document demo question:

```text
Kiracı ödeme yapmazsa kiraya veren ne yapabilir?
```

---

### 2. Instructor Benchmark Notebook

```text
notebooks/Demo System BenchmarkRAG.ipynb
```

This notebook:

1. Clones the GitHub repository
2. Installs requirements
3. Downloads the models from Hugging Face
4. Verifies model files
5. Runs the lightweight smoke test
6. Creates a demo benchmark JSON file
7. Launches the instructor benchmark UI

The benchmark UI supports two modes:

| Mode                     | Description                                                                             |
| ------------------------ | --------------------------------------------------------------------------------------- |
| Retrieval-only Benchmark | Computes Recall@k and MRR without loading the generator                                 |
| Full RAG Benchmark       | Runs retrieval and answer generation; can compute answer-level metrics such as Token F1 |

For stable remote evaluation, the retrieval-only benchmark mode is recommended first because it uses less GPU memory.

---

## Google Colab Instructions

For both notebooks:

1. Open the notebook in Google Colab.
2. Select:

```text
Runtime → Change runtime type → GPU
```

3. Run all cells from top to bottom.
4. Wait until the smoke test prints:

```text
LIGHT SMOKE TEST OK
```

5. Use the UI shown in the final cell.

The notebooks are designed so that the instructor can run the system by clicking the cells in order.

---

## Local Setup

Local setup is optional. GPU is strongly recommended.

Clone the repository:

```bash
git clone https://github.com/MberkKeskin/turkish_legal_rag_assistant.git
cd turkish_legal_rag_assistant/FINAL_SUBMISSION
```

Install requirements:

```bash
pip install -r requirements.txt
```

Download model files from Hugging Face into `FINAL_SUBMISSION/models/`.

Then run:

```bash
python smoke_test_light.py
```

To launch the UI manually:

```python
exec(open("final_ui.py", encoding="utf-8").read())
```

---

## Smoke Tests

### Lightweight Smoke Test

```bash
python smoke_test_light.py
```

This test checks the retrieval-side final pipeline without loading the full generator. It is recommended for quick validation and lower GPU memory usage.

Expected output:

```text
LIGHT SMOKE TEST OK
```

### Full Smoke Test

```bash
python smoke_test.py
```

This test loads the full final pipeline including the generator. It requires more GPU memory.

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

Expected benchmark JSON format:

```json
[
  {
    "question": "Türk Borçlar Kanunu m.314 kapsamında ifa zamanı nasıl düzenlenmiştir?",
    "gold_answer": "Kiracı, aksine sözleşme ve yerel adet olmadıkça kira bedelini ve gerekiyorsa yan giderleri her ayın sonunda ve en geç kira süresinin bitiminde ödemekle yükümlüdür.",
    "gold_ids": ["turkish_law_eski_6098_turk_borclar_kanunu_m314"]
  }
]
```

If `gold_ids` are provided, retrieval metrics such as Recall@k and MRR can be calculated.

If `gold_answer` is provided and Full RAG mode is used, answer-level metrics such as Token F1 can also be calculated.

---

## Evaluation Results

Evaluation outputs are stored under:

```text
FINAL_SUBMISSION/evaluation_results/
```

Important stagewise evaluation files:

```text
FINAL_SUBMISSION/evaluation_results/physical_stagewise_stratified50/physical_stagewise_summary.csv
FINAL_SUBMISSION/evaluation_results/physical_stagewise_stratified50/physical_stagewise_details.csv
```

The compared physical stages are:

| Stage                   | Description                                             |
| ----------------------- | ------------------------------------------------------- |
| S0_BASE                 | Base embedding + earliest available reranker + base LLM |
| S1_FT_EMBEDDING         | Fine-tuned embedding added                              |
| S2_FT_RERANKER          | Fine-tuned reranker added                               |
| S3_ERROR_MINED_RERANKER | Error-mined reranker and faithful LLM                   |
| S4_FINAL_SYSTEM         | Final system with default_top5                          |

---

## System Components

### Embedding Retriever

The embedding retriever maps Turkish legal questions and legal text chunks into the same vector space.

Final model:

```text
models/bge-m3-legal-ft-system2
```

This model was fine-tuned for Turkish legal retrieval so that questions and relevant legal passages are closer in embedding space.

---

### Reranker

The reranker scores retrieved candidate passages more precisely against the user question.

Final model:

```text
models/bge_reranker_legal_ft_v4_error_mined_hf
```

The final reranker was improved through multiple training stages, including error-mined examples. Error mining was used to focus the reranker on difficult cases where previous retrieval versions confused similar legal passages.

---

### Generator

The final answer generator uses a Qwen2.5-3B-Instruct base model with a legal LoRA fine-tuning adapter.

Final adapter:

```text
models/qwen2_5_3b_legal_lora_sft_faithful_v2_final
```

The generator is instructed to produce source-grounded Turkish legal answers and avoid unsupported claims.

---



## Notes and Limitations

* GPU runtime is recommended.
* Full RAG mode uses more memory because it loads the generator.
* Retrieval-only benchmark mode is safer for quick instructor-side evaluation.
* Token F1 may underestimate semantically correct answers that use different wording from the gold answer.
* Long legal decision texts can still be difficult because many chunks may be semantically similar.
* The UI is notebook-based rather than a deployed standalone web application.

---

## Quick Instructor Workflow

Recommended remote evaluation workflow:

```text
1. Open notebooks/Demo QuestionAnswerRAG.ipynb in Google Colab.
2. Enable GPU runtime.
3. Run all cells.
4. Ask a legal question in the final UI.
5. Open notebooks/benchmark_demo_click_run.ipynb in a separate Colab session.
6. Enable GPU runtime.
7. Run all cells.
8. Upload a benchmark JSON or use the built-in demo benchmark.
9. Run retrieval-only benchmark first.
```

This setup allows the instructor to test the project remotely without manually downloading model files.
