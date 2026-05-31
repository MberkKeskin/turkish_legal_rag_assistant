from pathlib import Path
import csv
import json
import re


def _normalize_text(text: str) -> str:
    text = str(text).replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_txt_files(data_dir: Path) -> list[dict]:
    docs: list[dict] = []
    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        docs.append(
            {
                "id": path.stem,
                "text": text,
                "source": path.name,
            }
        )
    return docs


def load_legal_corpus(csv_path: Path, limit: int | None = None) -> list[dict]:
    """
    Load UNIQUE legal contexts from the CSV dataset.
    Important: this dataset repeats the same context across many QA rows,
    so we deduplicate by normalized context text.
    """
    docs: list[dict] = []
    seen_contexts: set[str] = set()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            context = _normalize_text(row.get("context", ""))
            if not context:
                continue

            if context in seen_contexts:
                continue
            seen_contexts.add(context)

            docs.append(
                {
                    "id": f"legal_{len(docs)}",
                    "text": context,
                    "source": row.get("kaynak", "").strip() or "unknown",
                    "data_type": row.get("veri türü", "").strip() or "unknown",
                }
            )

            if limit is not None and len(docs) >= limit:
                break

    return docs


def load_legal_qa_pairs(csv_path: Path, limit: int | None = None) -> list[dict]:
    pairs: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            question = _normalize_text(row.get("soru", ""))
            answer = _normalize_text(row.get("cevap", ""))
            if not question:
                continue
            pairs.append(
                {
                    "question": question,
                    "answer": answer,
                    "source": row.get("kaynak", "").strip() or "unknown",
                    "data_type": row.get("veri türü", "").strip() or "unknown",
                    "id": f"legal_qa_{idx}",
                }
            )
    return pairs


def load_legal_hf_qa_pairs(data_dir: Path, limit: int | None = None) -> list[dict]:
    pairs: list[dict] = []
    if not data_dir.exists():
        return pairs

    json_files = [
        data_dir / "train.json",
        data_dir / "test.json",
        data_dir / "main set.json",
    ]

    def _extract_fields(item: dict) -> tuple[str, str]:
        question = item.get("Soru") or item.get("soru") or item.get("question") or ""
        answer = item.get("Cevap") or item.get("cevap") or item.get("answer") or ""
        return _normalize_text(question), _normalize_text(answer)

    for path in json_files:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            if limit is not None and len(pairs) >= limit:
                return pairs
            question, answer = _extract_fields(item)
            if not question:
                continue
            pairs.append(
                {
                    "question": question,
                    "answer": answer,
                    "source": path.name,
                    "data_type": "hf_legal_qa",
                    "id": f"hf_qa_{len(pairs)}",
                }
            )
    return pairs


def load_strict_corpus(path):
    """
    Loads strict verified JSONL corpus.

    Expected JSONL row format:
    {
        "id": "...",
        "text": "...",
        "title": "...",
        "metadata": {...}
    }

    Returns:
        List[dict] with keys:
        id, text, source, title, metadata
    """
    import json
    from pathlib import Path

    path = Path(path)
    docs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            doc_id = str(row.get("id", "")).strip()
            text = str(row.get("text", "")).strip()
            title = str(row.get("title", "")).strip()
            metadata = row.get("metadata", {}) or {}

            if not doc_id or not text:
                continue

            source = (
                metadata.get("law_name")
                or metadata.get("source")
                or metadata.get("citation_label")
                or title
                or "unknown"
            )

            docs.append({
                "id": doc_id,
                "chunk_id": doc_id,
                "text": text,
                "source": source,
                "title": title,
                "metadata": metadata,
            })

    return docs


def load_gold_benchmark_240(path):
    """
    Loads 240-item gold benchmark JSON file.
    """
    import json
    from pathlib import Path

    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rag_eval_1000(path):
    """
    Loads 1000-item RAG eval JSON file.
    """
    import json
    from pathlib import Path

    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
