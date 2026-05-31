
import sys, json
from pathlib import Path

BASE_DIR = Path('/content/drive/MyDrive/turkish-legal-rag-pipeline-feat-baseline-rag-arda/baseline_rag')
sys.path.append(str(BASE_DIR))

from app.final_pipeline.final_rag_pipeline import FinalLegalRAGPipeline

question = "Türk Borçlar Kanunu m.314 kapsamında ifa zamanı nasıl düzenlenmiştir?"

pipeline = FinalLegalRAGPipeline(
    base_dir=str(BASE_DIR),
    load_generator=False,
    candidate_top_k=80,
    rerank_top_k=5,
)

pipeline.load()

contexts = pipeline.retrieve_contexts(question)
serialized_contexts = [pipeline.serialize_context(c) for c in contexts]

payload = {
    "question": question,
    "contexts": serialized_contexts,
    "retrieved_ids": [c["id"] for c in serialized_contexts],
}

out_path = Path('/content/drive/MyDrive/turkish-legal-rag-pipeline-feat-baseline-rag-arda/baseline_rag/testResults/tmp_end_to_end_smoke_context.json')
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print("RETRIEVAL TEST OK")
print("Saved:", out_path)
print("Retrieved IDs:", payload["retrieved_ids"])
