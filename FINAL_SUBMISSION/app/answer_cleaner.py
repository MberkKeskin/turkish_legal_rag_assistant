
import re


def clean_prediction(pred: str) -> str:
    """
    Cleans generated LLM output for evaluation and final display.

    The fine-tuned model sometimes produces sections such as:
    Kaynak:, Metin:, Cevap:, Note:, Uyarı:, Soru:
    This function keeps the most answer-like grounded part.
    """
    pred = str(pred).strip()

    if re.search(r"\bassistant\b", pred, flags=re.IGNORECASE):
        parts = re.split(r"\bassistant\b", pred, flags=re.IGNORECASE)
        pred = parts[-1].strip()

    pred = re.split(r"\buser\b", pred, flags=re.IGNORECASE)[0].strip()
    pred = re.split(r"\bsystem\b", pred, flags=re.IGNORECASE)[0].strip()

    pred = re.split(r"\n\s*Soru\s*:", pred, flags=re.IGNORECASE)[0].strip()

    bad_prefixes = [
        r"Özür dilerim,.*?\n",
        r"Uyarı\s*:.*?\n",
        r"Note\s*:.*?\n",
        r"Kaynakça.*?\n",
    ]

    for pattern in bad_prefixes:
        pred = re.sub(pattern, "", pred, flags=re.IGNORECASE | re.DOTALL).strip()

    pred = pred.replace("Metin according to the source:", "Metin:")
    pred = pred.replace("metin according to the source:", "Metin:")
    pred = pred.replace("according to the source:", "")
    pred = pred.replace("According to the source:", "")

    metin_match = re.search(
        r"Metin\s*:\s*(.*?)(?:\n\s*Cevap\s*:|\n\s*Kaynak\s*:|\Z)",
        pred,
        flags=re.IGNORECASE | re.DOTALL
    )

    if metin_match:
        metin = metin_match.group(1).strip()
        if len(metin.split()) >= 8:
            pred = metin
    else:
        cevap_match = re.search(
            r"Cevap\s*:\s*(.*)",
            pred,
            flags=re.IGNORECASE | re.DOTALL
        )

        if cevap_match:
            cevap = cevap_match.group(1).strip()
            if len(cevap.split()) >= 4:
                pred = cevap

    pred = re.sub(r"\n\s*Kaynak(?:/Citation|çağı|ı)?\s*:.*", "", pred, flags=re.IGNORECASE).strip()
    pred = re.sub(r"\n\s*Citation\s*:.*", "", pred, flags=re.IGNORECASE).strip()

    pred = re.sub(r"\n{3,}", "\n\n", pred)
    pred = re.sub(r"[ \t]+", " ", pred).strip()

    return pred
