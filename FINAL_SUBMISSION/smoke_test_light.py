
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.final_pipeline.final_v8_pipeline import FinalLegalRAGPipelineV8


def main():
    print("Loading final legal RAG pipeline without generator...")

    pipeline = FinalLegalRAGPipelineV8(
        base_dir=str(BASE_DIR),
        load_generator=False,
        candidate_top_k=100,
        rerank_top_k=5,
        max_expanded_candidates=300,
    )

    pipeline.load()

    question = "Türk Borçlar Kanunu m.314 kapsamında ifa zamanı nasıl düzenlenmiştir?"

    print("\nQuestion:")
    print(question)

    contexts = pipeline.retrieve_contexts(question)
    ids = []
    for c in contexts:
        ids.append(c.get("id") or c.get("chunk_id"))

    print("\nRetrieved IDs:")
    print(ids)

    print("\nLIGHT SMOKE TEST OK")


if __name__ == "__main__":
    main()
