import re


def basic_chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Chunk text by fixed character windows with overlap."""
    chunks: list[str] = []
    start = 0
    text = text.strip()
    if not text:
        return chunks
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def sentence_chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Chunk text by sentences while respecting a max character size."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    raw_chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                raw_chunks.append(current)
            current = sentence
    if current:
        raw_chunks.append(current)

    if overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped: list[str] = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            overlapped.append(chunk)
            continue
        tail = raw_chunks[i - 1][-overlap:]
        overlapped.append((tail + " " + chunk).strip())
    return overlapped


def legal_chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Chunk legal text by detecting article headings like MADDE/Madde/md."""
    headings = re.split(r"(?=(?:\bMADDE\b|\bMadde\b|\bmd\.\b)\s*\d+)", text)
    sections = [h.strip() for h in headings if h.strip()]
    if not sections:
        return sentence_chunk_text(text, chunk_size, overlap)

    chunks: list[str] = []
    for section in sections:
        chunks.extend(sentence_chunk_text(section, chunk_size, overlap))
    return chunks


def chunk_documents(
    docs: list[dict],
    chunk_size: int,
    overlap: int,
    use_sentence_chunking: bool = True,
    use_legal_chunking: bool = False,
) -> list[dict]:
    """Chunk documents into smaller pieces with metadata."""
    all_chunks: list[dict] = []
    for doc in docs:
        if use_legal_chunking:
            chunks = legal_chunk_text(doc["text"], chunk_size, overlap)
        elif use_sentence_chunking:
            chunks = sentence_chunk_text(doc["text"], chunk_size, overlap)
        else:
            chunks = basic_chunk_text(doc["text"], chunk_size, overlap)
        for idx, chunk in enumerate(chunks):
            chunk_item = {
                "chunk_id": f"{doc['id']}_chunk_{idx}",
                "text": chunk,
                "source": doc["source"],
            }
            for key, value in doc.items():
                if key in {"id", "text", "source"}:
                    continue
                chunk_item[key] = value
            all_chunks.append(chunk_item)
    return all_chunks
