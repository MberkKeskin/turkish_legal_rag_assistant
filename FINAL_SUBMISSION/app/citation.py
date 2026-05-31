
def get_result_id(item: dict) -> str:
    """
    Robustly extracts a chunk/result identifier from a retrieved result.
    """
    return (
        item.get("id")
        or item.get("chunk_id")
        or item.get("corpus_row_id")
        or item.get("source_id")
        or ""
    )


def build_citation(item: dict) -> str:
    """
    Builds deterministic citation string from retrieved chunk.
    """
    rid = get_result_id(item)
    source = item.get("source", "")
    title = item.get("title", "")

    parts = []
    if source:
        parts.append(str(source))
    if title and title != source:
        parts.append(str(title))
    if rid:
        parts.append(str(rid))

    return " - ".join(parts)


def citation_accuracy(prediction: str, retrieved_items: list) -> float:
    """
    Checks whether generated prediction contains one of the retrieved source IDs or source names.
    """
    pred = str(prediction).lower()

    for item in retrieved_items:
        rid = str(get_result_id(item)).lower()
        source = str(item.get("source", "")).lower()

        if rid and rid in pred:
            return 1.0

        if source and source in pred:
            return 1.0

    return 0.0
