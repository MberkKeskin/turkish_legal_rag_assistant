
from app.answer_cleaner import clean_prediction
from app.citation import build_citation, get_result_id
from app.extractive_guard import guarded_extractive_answer


def build_context_from_retrieved(retrieved, max_context_chars=2200, max_chunks=1):
    parts = []
    total = 0

    for item in retrieved[:max_chunks]:
        text = str(item.get("text", "")).strip()
        source = str(item.get("source", ""))
        rid = str(get_result_id(item))

        block = f"[ID: {rid}]\n[SOURCE: {source}]\n{text}"

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


def build_ft_messages(question, context):
    system_prompt = (
        "Sen bir Türk hukuku RAG asistanısın. "
        "Yalnızca kullanıcı tarafından verilen kaynak metne dayanarak cevap ver. "
        "Kaynakta olmayan bilgiyi üretme. "
        "Kanun maddesi soruluyorsa ilgili madde metnini mümkün olduğunca aynen aktar. "
        "Cevabın sonunda kaynak/citation bilgisini belirt."
    )

    user_prompt = f"""
[Kaynak]
{context}

Soru: {question}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_llm_answer(question, retrieved, model, tokenizer, max_new_tokens=180, max_context_chars=2200, max_chunks=1):
    context = build_context_from_retrieved(
        retrieved,
        max_context_chars=max_context_chars,
        max_chunks=max_chunks
    )

    messages = build_ft_messages(question, context)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned = clean_prediction(decoded)

    if retrieved:
        citation = build_citation(retrieved[0])
        if "Kaynak" not in cleaned and citation:
            cleaned = f"{cleaned}\n\nKaynak: {citation}"

    return cleaned, decoded, context


def answer_llm_only(question, retrieved, model, tokenizer):
    answer, raw, context = generate_llm_answer(
        question=question,
        retrieved=retrieved,
        model=model,
        tokenizer=tokenizer,
        max_chunks=1
    )

    return {
        "answer": answer,
        "raw_answer": raw,
        "context": context,
        "mode": "llm_only",
        "used_guard": False,
    }


def answer_guarded(question, retrieved, model, tokenizer):
    guarded_answer, used_guard, selected_item = guarded_extractive_answer(question, retrieved)

    context = build_context_from_retrieved(retrieved, max_chunks=1)

    if used_guard:
        return {
            "answer": guarded_answer,
            "raw_answer": guarded_answer,
            "context": context,
            "mode": "extractive_guard",
            "used_guard": True,
        }

    answer, raw, context = generate_llm_answer(
        question=question,
        retrieved=retrieved,
        model=model,
        tokenizer=tokenizer,
        max_chunks=1
    )

    return {
        "answer": answer,
        "raw_answer": raw,
        "context": context,
        "mode": "llm_fallback",
        "used_guard": False,
    }
