
from pathlib import Path
from typing import List, Dict

from app.custom_document_loader import load_and_chunk_custom_documents
from app.bm25_retriever import BM25Retriever, get_result_id


class CustomRagPipeline:
    """
    RAG pipeline for user-provided custom documents.
    It builds a temporary BM25 index over PDF/TXT/MD documents.
    """

    def __init__(
        self,
        docs_dir: str | Path,
        chunk_size: int = 900,
        overlap: int = 150
    ):
        self.docs_dir = Path(docs_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.chunks: List[Dict] = []
        self.retriever: BM25Retriever | None = None

    def build_index(self):
        self.chunks = load_and_chunk_custom_documents(
            self.docs_dir,
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )

        if not self.chunks:
            raise ValueError(
                f"No supported documents found in {self.docs_dir}. "
                "Supported formats: .pdf, .txt, .md"
            )

        self.retriever = BM25Retriever(self.chunks)

    def retrieve(self, query: str, top_k: int = 5):
        if self.retriever is None:
            raise RuntimeError("Index is not built. Call build_index() first.")

        return self.retriever.retrieve(query, top_k=top_k)

    def build_context(self, retrieved, max_context_chars: int = 3000, max_chunks: int = 3):
        parts = []
        total = 0

        for item in retrieved[:max_chunks]:
            source = item.get("source", "unknown")
            text = str(item.get("text", "")).strip()

            if not text:
                continue

            block = f"[SOURCE: {source}]\n{text}"

            remaining = max_context_chars - total
            if remaining <= 0:
                break

            if len(block) <= remaining:
                parts.append(block)
                total += len(block)
            else:
                parts.append(block[:remaining])
                total += remaining
                break

        return "\n\n".join(parts)


def build_custom_rag_messages(question: str, context: str):
    system_prompt = (
        "Sen verilen dokümanlara dayalı cevap veren bir asistansın. "
        "Sadece sağlanan bağlamdaki bilgileri kullan. "
        "Bağlamda cevap yoksa 'Bilmiyorum' yaz."
    )

    user_prompt = f"""
BAĞLAM:
{context}

SORU:
{question}

KURALLAR:
- Sadece BAĞLAM içindeki bilgiye dayan.
- Cevabı Türkçe ver.
- Belgedeki ifadeyi mümkün olduğunca koru.
- Uydurma bilgi ekleme.
- Cevap bağlamda yoksa sadece "Bilmiyorum" yaz.
- Cevabın sonunda kullandığın kaynak dosya adını belirt.

CEVAP:
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_answer(generator, messages, max_new_tokens: int = 256):
    tokenizer = generator.tokenizer
    model = generator.model
    device = generator.device

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "CEVAP:" in decoded:
        answer = decoded.split("CEVAP:")[-1].strip()
    else:
        answer = decoded.strip()

    answer = answer.split("SORU:")[0].strip()
    answer = answer.split("BAĞLAM:")[0].strip()
    answer = answer.split("KURALLAR:")[0].strip()

    return answer
