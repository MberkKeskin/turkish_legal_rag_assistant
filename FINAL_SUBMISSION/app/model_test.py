from .config import (
    GENERATION_MODEL_NAME,
    GENERATION_MODEL_PATH,
    GENERATION_LOCAL_ONLY,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_LOCAL_ONLY,
    EMBEDDING_TRUST_REMOTE_CODE,
    EMBEDDING_QUERY_PREFIX,
    EMBEDDING_DOCUMENT_PREFIX,
)
from .generator import Generator
from .embedder import Embedder


def main() -> None:
    print("Embedding model name:", EMBEDDING_MODEL_NAME)
    print("Embedding local path:", EMBEDDING_MODEL_PATH or "(not set)")
    print("Embedding local-only mode:", EMBEDDING_LOCAL_ONLY)

    embedder = Embedder(
        EMBEDDING_MODEL_NAME,
        model_path=EMBEDDING_MODEL_PATH,
        local_only=EMBEDDING_LOCAL_ONLY,
        trust_remote_code=EMBEDDING_TRUST_REMOTE_CODE,
        query_prefix=EMBEDDING_QUERY_PREFIX,
        document_prefix=EMBEDDING_DOCUMENT_PREFIX,
    )

    print("Resolved embedding source:", embedder.source)
    print("Embedding device:", embedder.device)

    sample_q = ["Sözleşmenin temel unsurları nelerdir?"]
    sample_d = ["Sözleşmenin temel unsurları; tarafların karşılıklı ve birbirine uygun irade beyanı, sözleşmenin konusu ve hukuka uygun amaçtır."]

    q_emb = embedder.encode_queries(sample_q)
    d_emb = embedder.encode_documents(sample_d)

    print("Query embedding shape:", q_emb.shape)
    print("Doc embedding shape:", d_emb.shape)

    print()
    print("Selected generation model:", GENERATION_MODEL_NAME)
    print("Local generation path:", GENERATION_MODEL_PATH or "(not set)")
    print("Generation local-only mode:", GENERATION_LOCAL_ONLY)

    generator = Generator(
        GENERATION_MODEL_NAME,
        model_path=GENERATION_MODEL_PATH,
        local_only=GENERATION_LOCAL_ONLY,
    )

    print("Resolved generation source:", generator.source)
    print("Generation device:", generator.device)

    context = (
        "Sözleşmenin temel unsurları; tarafların karşılıklı ve birbirine uygun irade beyanı, "
        "sözleşmenin konusu ve hukuka uygun amaçtır."
    )
    prompt = (
        "Görev: Türk hukukuna dair soruyu yalnızca verilen bağlamla yanıtla. "
        "Bağlam dışı genel bilgi kullanma. "
        'Bağlamda yoksa sadece "Bilmiyorum" yaz. '
        "Ek bilgi, yorum veya varsayım ekleme. "
        "Yanıt kısa ve doğrudan olsun.\n\n"
        f"Bağlam:\n{context}\n\n"
        "Soru: Türk hukukunda sözleşmenin temel unsurları nelerdir?\nCevap:"
    )
    answer = generator.generate(prompt, max_new_tokens=80)
    print("Output:", answer)


if __name__ == "__main__":
    main()