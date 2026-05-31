
import sys
import re
import io
import time
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.final_pipeline.final_v8_pipeline import FinalLegalRAGPipelineV8


# ============================================================
# STYLE
# ============================================================

def inject_css():
    display(HTML("""
    <style>
        .legal-app {
            font-family: Inter, Arial, sans-serif;
            max-width: 1080px;
            margin: 0 auto;
        }

        .hero {
            background: linear-gradient(135deg, #111827 0%, #1e40af 50%, #2563eb 100%);
            color: white;
            border-radius: 26px;
            padding: 32px 36px;
            margin: 18px 0 22px 0;
            box-shadow: 0 18px 40px rgba(15,23,42,0.28);
        }

        .hero-title {
            font-size: 36px;
            font-weight: 850;
            margin: 0;
            letter-spacing: -0.5px;
        }

        .hero-subtitle {
            font-size: 16px;
            margin: 12px 0 0 0;
            line-height: 1.65;
            opacity: 0.94;
            max-width: 850px;
        }

        .panel {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 22px;
            padding: 22px 24px;
            margin: 18px 0;
            box-shadow: 0 10px 28px rgba(15,23,42,0.08);
        }

        .panel-title {
            font-size: 24px;
            font-weight: 800;
            color: #0f172a;
            margin: 0 0 8px 0;
        }

        .panel-desc {
            color: #475569;
            font-size: 15px;
            line-height: 1.6;
            margin: 0;
        }

        .question-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 16px 18px;
            margin: 14px 0 12px 0;
            color: #334155;
        }

        .question-card b {
            color: #0f172a;
        }

        .timeline {
            margin: 16px 0;
        }

        .step {
            display: flex;
            gap: 14px;
            align-items: center;
            padding: 14px 16px;
            margin: 9px 0;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            color: #334155;
            font-size: 15px;
        }

        .step.active {
            background: #eff6ff;
            border-color: #93c5fd;
            color: #1e3a8a;
        }

        .step.done {
            background: #ecfdf5;
            border-color: #86efac;
            color: #065f46;
        }

        .step-icon {
            min-width: 30px;
            width: 30px;
            height: 30px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #cbd5e1;
            color: white;
            font-weight: 800;
            font-size: 14px;
        }

        .step.active .step-icon {
            background: #2563eb;
        }

        .step.done .step-icon {
            background: #10b981;
        }

        .answer {
            background: linear-gradient(180deg, #ecfdf5 0%, #f0fdf4 100%);
            border: 1px solid #10b981;
            border-radius: 22px;
            padding: 24px 26px;
            margin: 20px 0;
            box-shadow: 0 12px 30px rgba(16,185,129,0.14);
        }

        .answer h2 {
            color: #065f46;
            font-size: 26px;
            margin: 0 0 14px 0;
        }

        .answer-text {
            color: #064e3b;
            font-size: 16px;
            line-height: 1.8;
            white-space: pre-wrap;
        }

        .sources-title {
            color: #0f172a;
            font-size: 22px;
            font-weight: 800;
            margin: 22px 0 10px 0;
        }

        .source {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 18px 20px;
            margin: 12px 0;
            box-shadow: 0 6px 18px rgba(15,23,42,0.06);
        }

        .source-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 10px;
        }

        .source-title {
            color: #111827;
            font-weight: 800;
            font-size: 16px;
        }

        .source-badge {
            background: #e0f2fe;
            color: #075985;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 12px;
            font-weight: 700;
        }

        .source-id {
            color: #94a3b8;
            font-size: 11px;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            margin-bottom: 8px;
        }

        .source-text {
            background: #f8fafc;
            border-radius: 14px;
            padding: 14px 15px;
            color: #334155;
            line-height: 1.62;
            font-size: 14px;
            white-space: pre-wrap;
            max-height: 220px;
            overflow-y: auto;
        }

        .info-strip {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 15px 17px;
            margin: 14px 0;
            color: #475569;
            font-size: 14px;
            line-height: 1.6;
        }

        .mini-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 15px 0;
        }

        .mini-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 16px;
        }

        .mini-card .big {
            font-size: 22px;
            font-weight: 850;
            color: #0f172a;
            margin-bottom: 4px;
        }

        .mini-card .label {
            color: #64748b;
            font-size: 13px;
        }

        .warning {
            background: #fff7ed;
            border: 1px solid #fdba74;
            color: #9a3412;
            padding: 14px 16px;
            border-radius: 16px;
            margin: 12px 0;
        }

        .success-small {
            background: #ecfdf5;
            border: 1px solid #86efac;
            color: #065f46;
            padding: 14px 16px;
            border-radius: 16px;
            margin: 12px 0;
        }
    </style>
    """))


def wrap(html):
    display(HTML(f"<div class='legal-app'>{html}</div>"))


def esc(text):
    text = str(text)
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


def hero():
    wrap("""
    <div class="hero">
        <div class="hero-title">⚖️ Turkish Legal RAG Assistant</div>
        <div class="hero-subtitle">
            Hukuki sorular için kaynaklara dayalı cevap üretir. Sistem önce ilgili belgeleri bulur,
            sonra en uygun kaynakları seçer ve cevabı bu kaynaklara dayanarak hazırlar.
        </div>
    </div>
    """)


def panel(title, desc):
    wrap(f"""
    <div class="panel">
        <div class="panel-title">{title}</div>
        <p class="panel-desc">{desc}</p>
    </div>
    """)


def question_box(question, file_name=None):
    if file_name:
        extra = f"<br><b>Yüklenen belge:</b> {esc(file_name)}"
    else:
        extra = ""

    wrap(f"""
    <div class="question-card">
        <b>Soru:</b> {esc(question)}
        {extra}
    </div>
    """)


def step(num, text, status="active"):
    wrap(f"""
    <div class="step {status}">
        <div class="step-icon">{num}</div>
        <div>{esc(text)}</div>
    </div>
    """)


def answer_card(answer):
    wrap(f"""
    <div class="answer">
        <h2>✅ Cevap</h2>
        <div class="answer-text">{esc(answer)}</div>
    </div>
    """)


def warning(msg):
    wrap(f"""
    <div class="warning">⚠️ {esc(msg)}</div>
    """)


def success(msg):
    wrap(f"""
    <div class="success-small">✅ {esc(msg)}</div>
    """)


def summary_strip(found, used, mode):
    if mode == "legal":
        text = "Sistem, mevcut hukuk veri tabanında arama yaptı ve cevap için en uygun kaynakları seçti."
    else:
        text = "Sistem, yüklenen belgeyi parçalara ayırdı ve soruyla en ilgili bölümleri seçti."

    wrap(f"""
    <div class="info-strip">
        <b>İşlem Özeti:</b> {text}
        <div class="mini-grid">
            <div class="mini-card">
                <div class="big">{found}</div>
                <div class="label">Taranan / oluşturulan kaynak parçası</div>
            </div>
            <div class="mini-card">
                <div class="big">{used}</div>
                <div class="label">Cevap için kullanılan kaynak</div>
            </div>
            <div class="mini-card">
                <div class="big">Kaynaklı</div>
                <div class="label">Cevap üretim biçimi</div>
            </div>
        </div>
    </div>
    """)


def sources_card(contexts, title="Kullanılan Kaynaklar"):
    wrap(f"<div class='sources-title'>📚 {title}</div>")

    if not contexts:
        wrap("<div class='info-strip'>Gösterilecek kaynak bulunamadı.</div>")
        return

    for i, ctx in enumerate(contexts, 1):
        cid = ctx.get("id") or ctx.get("chunk_id") or f"source_{i}"
        text = ctx.get("text") or ctx.get("content") or ""

        wrap(f"""
        <div class="source">
            <div class="source-header">
                <div class="source-title">Kaynak {i}</div>
                <div class="source-badge">Cevapta kullanıldı</div>
            </div>
            <div class="source-id">{esc(cid)}</div>
            <div class="source-text">{esc(text[:2200])}</div>
        </div>
        """)


def normalize_tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-ZğüşöçıİĞÜŞÖÇ0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def simple_retrieve(question, chunks, top_k=5):
    q_tokens = set(normalize_tokenize(question))
    scored = []

    for c in chunks:
        token_set = set(normalize_tokenize(c["text"]))
        overlap = len(q_tokens & token_set)
        density = overlap / max(1, len(q_tokens))
        score = overlap + density

        x = dict(c)
        x["score"] = score
        scored.append(x)

    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


def chunk_text(text, chunk_size=900, overlap=150):
    text = re.sub(r"\s+", " ", str(text)).strip()
    chunks = []

    start = 0
    idx = 1

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "id": f"uploaded_doc_chunk_{idx}",
                "text": chunk,
                "source": "uploaded_document"
            })

        idx += 1

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def extract_uploaded_text(upload_widget):
    if not upload_widget.value:
        return None, None

    uploaded = list(upload_widget.value.values())[0]
    filename = uploaded["metadata"]["name"]
    content = uploaded["content"]

    lower = filename.lower()

    if lower.endswith(".txt"):
        return filename, content.decode("utf-8", errors="ignore")

    if lower.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join([(p.extract_text() or "") for p in reader.pages])
            return filename, text
        except Exception as e:
            raise RuntimeError(f"PDF okunamadı: {e}")

    if lower.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(content))
            text = "\n".join([p.text for p in doc.paragraphs])
            return filename, text
        except Exception as e:
            raise RuntimeError(f"DOCX okunamadı: {e}")

    raise RuntimeError("Desteklenmeyen dosya türü. TXT, PDF veya DOCX yükleyin.")


# ============================================================
# LOAD APP
# ============================================================

inject_css()
hero()

load_output = widgets.Output()
display(load_output)

with load_output:
    step("•", "Sistem hazırlanıyor...", "active")

    pipeline = FinalLegalRAGPipelineV8(
        base_dir=str(BASE_DIR),
        load_generator=True,
        candidate_top_k=100,
        rerank_top_k=5,
        max_expanded_candidates=300,
    )
    pipeline.load()

    clear_output()
    success("Sistem hazır. Soru sorabilir veya kendi belgenizi yükleyebilirsiniz.")


# ============================================================
# LEGAL KB MODE
# ============================================================

panel(
    "1️⃣ Hukuk Veri Tabanında Soru-Cevap",
    "Mevcut hukuk veri tabanı üzerinden soru sorun. Sistem ilgili kaynakları bulur, seçer ve cevabı kaynaklara dayalı olarak üretir."
)

examples = widgets.Dropdown(
    options=[
        "Kiracı kira bedelini her ay ne zaman ödemekle yükümlüdür?",
        "Türk Borçlar Kanunu m.314 kapsamında ifa zamanı nasıl düzenlenmiştir?",
        "Ceza muhakemesinde şüpheli ile sanık arasındaki fark nedir?",
        "Evli olmayan bir kişi tek başına evlat edinebilir mi? Varsa yaş şartı nedir?",
        "Kira sözleşmesinde kiracının ödeme borcu nasıl düzenlenmiştir?"
    ],
    description="Örnek:",
    layout=widgets.Layout(width="100%")
)

legal_question = widgets.Textarea(
    value=examples.value,
    placeholder="Hukuki sorunuzu yazın...",
    description="Soru:",
    layout=widgets.Layout(width="100%", height="95px")
)

def update_example(change):
    legal_question.value = change["new"]

examples.observe(update_example, names="value")

legal_button = widgets.Button(
    description="Cevap Üret",
    button_style="primary",
    layout=widgets.Layout(width="180px", height="42px")
)

legal_output = widgets.Output()


def on_legal_click(_):
    with legal_output:
        clear_output()

        q = legal_question.value.strip()

        if not q:
            warning("Lütfen bir soru yazın.")
            return

        question_box(q)

        step("1", "Soru alındı.", "done")
        time.sleep(0.2)

        step("2", "Hukuk kaynakları içinde arama yapılıyor.", "done")
        time.sleep(0.2)

        step("3", "Bulunan kaynaklar inceleniyor ve sıralanıyor.", "done")
        time.sleep(0.2)

        step("4", "Cevap için en uygun kaynaklar seçiliyor.", "done")
        time.sleep(0.2)

        step("5", "Cevap hazırlanıyor.", "active")

        try:
            out = pipeline.answer(q)

            clear_output()
            question_box(q)

            step("1", "Soru alındı.", "done")
            step("2", "Hukuk kaynakları içinde arama yapıldı.", "done")
            step("3", "Bulunan kaynaklar incelendi ve sıralandı.", "done")
            step("4", "Cevap için en uygun kaynaklar seçildi.", "done")
            step("5", "Cevap hazırlandı.", "done")

            answer_card(out.get("answer", ""))

            retrieved_count = len(out.get("retrieved_ids", []))
            used_count = len(out.get("llm_contexts", []))
            summary_strip(retrieved_count, used_count, "legal")

            sources_card(out.get("llm_contexts", []))

        except Exception as e:
            warning(f"Cevap üretilirken hata oluştu: {e}")


legal_button.on_click(on_legal_click)

display(examples)
display(legal_question)
display(legal_button)
display(legal_output)


# ============================================================
# UPLOADED DOCUMENT MODE
# ============================================================

panel(
    "2️⃣ Kendi Belgeniz Üzerinden Soru-Cevap",
    "TXT, PDF veya DOCX belge yükleyin. Sistem belgeyi parçalara ayırır, sorunuzla ilgili bölümleri bulur ve cevabı sadece bu belgeye dayanarak üretir."
)

upload_widget = widgets.FileUpload(
    accept=".txt,.pdf,.docx",
    multiple=False,
    description="Belge Yükle"
)

doc_question = widgets.Textarea(
    value="Bu belgeye göre temel hüküm veya sonuç nedir?",
    placeholder="Yüklediğiniz belgeye göre soru sorun...",
    description="Soru:",
    layout=widgets.Layout(width="100%", height="95px")
)

doc_button = widgets.Button(
    description="Belgeden Cevap Üret",
    button_style="success",
    layout=widgets.Layout(width="230px", height="42px")
)

doc_output = widgets.Output()


def on_doc_click(_):
    with doc_output:
        clear_output()

        q = doc_question.value.strip()

        if not upload_widget.value:
            warning("Lütfen önce TXT, PDF veya DOCX belge yükleyin.")
            return

        if not q:
            warning("Lütfen belge hakkında bir soru yazın.")
            return

        try:
            step("1", "Belge yüklendi.", "done")
            time.sleep(0.2)

            filename, text = extract_uploaded_text(upload_widget)

            step("2", "Belge metni çıkarılıyor.", "done")
            time.sleep(0.2)

            if not text or len(text.strip()) < 50:
                warning("Belgeden yeterli metin çıkarılamadı.")
                return

            step("3", "Belge anlamlı parçalara ayrılıyor.", "done")
            chunks = chunk_text(text)
            time.sleep(0.2)

            step("4", "Soruyla ilgili belge bölümleri bulunuyor.", "done")
            selected = simple_retrieve(q, chunks, top_k=5)
            time.sleep(0.2)

            step("5", "Cevap hazırlanıyor.", "active")

            answer, raw, context_text, used_retry = pipeline.generator.generate(
                question=q,
                contexts=selected,
                context_top_k=len(selected),
                retry=True,
            )

            clear_output()
            question_box(q, file_name=filename)

            step("1", "Belge yüklendi.", "done")
            step("2", "Belge metni çıkarıldı.", "done")
            step("3", f"Belge {len(chunks)} parçaya ayrıldı.", "done")
            step("4", "Soruyla ilgili belge bölümleri bulundu.", "done")
            step("5", "Cevap hazırlandı.", "done")

            answer_card(answer)

            summary_strip(len(chunks), len(selected), "uploaded")
            sources_card(selected, title="Kullanılan Belge Bölümleri")

        except Exception as e:
            warning(f"Belgeye dayalı cevap üretilirken hata oluştu: {e}")


doc_button.on_click(on_doc_click)

display(upload_widget)
display(doc_question)
display(doc_button)
display(doc_output)
