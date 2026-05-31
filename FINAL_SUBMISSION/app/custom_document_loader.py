
from pathlib import Path
import re
from typing import List, Dict

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def read_txt_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    if PdfReader is None:
        raise ImportError("pypdf is not installed. Install it with: pip install pypdf")

    reader = PdfReader(str(path))
    pages = []

    for i, page in enumerate(reader.pages, 1):
        page_text = page.extract_text() or ""
        page_text = clean_text(page_text)

        if page_text:
            pages.append(f"[PAGE {i}]\n{page_text}")

    return "\n\n".join(pages)


def load_custom_documents(docs_dir: str | Path) -> List[Dict]:
    docs_dir = Path(docs_dir)

    if not docs_dir.exists():
        raise FileNotFoundError(f"Custom docs directory not found: {docs_dir}")

    supported_exts = {".txt", ".md", ".pdf"}
    docs = []

    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in supported_exts:
            continue

        if path.suffix.lower() in {".txt", ".md"}:
            text = read_txt_file(path)
        elif path.suffix.lower() == ".pdf":
            text = read_pdf_file(path)
        else:
            continue

        text = clean_text(text)

        if not text:
            continue

        docs.append({
            "source": path.name,
            "source_path": str(path),
            "title": path.stem,
            "text": text,
            "metadata": {
                "file_name": path.name,
                "file_path": str(path),
                "file_type": path.suffix.lower().replace(".", ""),
            }
        })

    return docs


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = clean_text(text)

    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        # Cümlenin ortasında kesmemeye çalış
        if end < len(text):
            last_period = max(
                chunk.rfind("."),
                chunk.rfind("?"),
                chunk.rfind("!"),
                chunk.rfind("\n")
            )

            if last_period > chunk_size * 0.55:
                chunk = chunk[:last_period + 1].strip()
                end = start + last_period + 1

        if chunk:
            chunks.append(chunk)

        next_start = end - overlap
        if next_start <= start:
            next_start = start + chunk_size

        start = next_start

    return chunks


def chunk_custom_documents(
    docs: List[Dict],
    chunk_size: int = 900,
    overlap: int = 150
) -> List[Dict]:
    chunks = []

    for doc_idx, doc in enumerate(docs):
        text_chunks = chunk_text(
            doc["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk_idx, chunk in enumerate(text_chunks):
            chunk_id = f"custom_{doc_idx:04d}_{chunk_idx:04d}"

            chunks.append({
                "id": chunk_id,
                "chunk_id": chunk_id,
                "source": doc["source"],
                "title": doc.get("title", doc["source"]),
                "text": chunk,
                "metadata": {
                    **doc.get("metadata", {}),
                    "doc_index": doc_idx,
                    "chunk_index": chunk_idx,
                    "custom_document": True,
                }
            })

    return chunks


def load_and_chunk_custom_documents(
    docs_dir: str | Path,
    chunk_size: int = 900,
    overlap: int = 150
) -> List[Dict]:
    docs = load_custom_documents(docs_dir)
    return chunk_custom_documents(
        docs,
        chunk_size=chunk_size,
        overlap=overlap
    )
