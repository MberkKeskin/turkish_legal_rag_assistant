
import re
from app.citation import get_result_id, build_citation


LAW_HINTS = {
    "türk borçlar kanunu": "borclar",
    "borçlar kanunu": "borclar",
    "türk medeni kanunu": "medeni",
    "medenî kanunu": "medeni",
    "medeni kanunu": "medeni",
    "ceza muhakemesi kanunu": "ceza_muhakemesi",
    "cmk": "ceza_muhakemesi",
    "anayasa": "anayasa",
    "türkiye cumhuriyeti anayasası": "anayasa",
    "bilgi edinme hakkı kanunu": "bilgi_edinme",
    "türk bayrağı tüzüğü": "bayragi",
}


def extract_article_number(question: str):
    """
    Extracts legal article number from Turkish question.
    Handles forms such as:
    m.314, m314, madde 314, Madde 314
    """
    q = str(question).lower()

    patterns = [
        r"\bm\.\s*(\d+)",
        r"\bm\s*(\d+)",
        r"\bmadde\s*(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return match.group(1)

    return None


def detect_law_hint(question: str):
    q = str(question).lower()

    for key, val in LAW_HINTS.items():
        if key in q:
            return val

    return None


def is_exact_article_match(question: str, item: dict) -> bool:
    """
    Determines whether retrieved chunk exactly matches requested law/article.
    """
    article_no = extract_article_number(question)
    law_hint = detect_law_hint(question)

    if not article_no:
        return False

    rid = get_result_id(item).lower()
    text = str(item.get("text", "")).lower()
    source = str(item.get("source", "")).lower()
    title = str(item.get("title", "")).lower()

    article_match = (
        f"_m{article_no}" in rid
        or f"madde {article_no}" in text
        or f"madde {article_no}-" in text
        or f"madde {article_no}–" in text
        or f"madde {article_no} -" in text
        or f"madde {article_no} –" in text
    )

    if not article_match:
        return False

    if law_hint:
        joined = " ".join([rid, source, title])
        return law_hint in joined

    return True


def find_exact_article_chunk(question: str, retrieved_items: list):
    for item in retrieved_items:
        if is_exact_article_match(question, item):
            return item
    return None


def extractive_answer_from_chunk(item: dict) -> str:
    """
    Returns the retrieved legal article text with deterministic citation.
    """
    text = str(item.get("text", "")).strip()
    citation = build_citation(item)

    return f"{text}\n\nKaynak: {citation}"


def guarded_extractive_answer(question: str, retrieved_items: list):
    """
    Returns (answer, used_guard, selected_item)
    """
    item = find_exact_article_chunk(question, retrieved_items)

    if item is None:
        return None, False, None

    return extractive_answer_from_chunk(item), True, item
