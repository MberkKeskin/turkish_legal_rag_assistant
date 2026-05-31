import re
from typing import Any, Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class QwenLegalGenerator:
    """
    Faithful Qwen generator.

    Final policy:
    - LLM always generates the answer.
    - No extractive fallback.
    - If answer is ID-only or too short, retry with stricter prompt.
    """

    def __init__(
        self,
        lora_dir: str,
        base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        max_context_chars: int = 5000,
        max_input_length: int = 4096,
        max_new_tokens: int = 260,
    ):
        self.lora_dir = str(lora_dir)
        self.base_model_name = base_model_name
        self.max_context_chars = max_context_chars
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens

        self.tokenizer = None
        self.base_model = None
        self.model = None

    def load(self) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        self.model = PeftModel.from_pretrained(
            self.base_model,
            self.lora_dir,
        )

        self.model.eval()

    def format_contexts(self, contexts: List[Dict[str, Any]], context_top_k: int = 5) -> str:
        parts = []
        used = 0

        for i, c in enumerate(contexts[:context_top_k], 1):
            cid = c.get("id", "")
            source = c.get("source", "")
            title = c.get("title", "")
            text = c.get("text", "")

            block = f"""[KAYNAK {i}]
ID: {cid}
Kaynak: {source}
Başlık: {title}
Metin:
{text}""".strip()

            if used + len(block) > self.max_context_chars:
                break

            parts.append(block)
            used += len(block)

        return "\n\n".join(parts)

    def build_messages(self, question: str, context_text: str):
        system_prompt = (
            "Sen bir Türk hukuku RAG asistanısın.\n"
            "Verilen kaynaklar arasından soruya en doğrudan cevap veren kaynağı belirle.\n"
            "Cevabı yalnızca seçtiğin kaynak veya kaynaklardaki bilgiye dayanarak üret.\n"
            "Kaynak dışı bilgi ekleme.\n"
            "Sadece kaynak ID yazma; mutlaka maddi cevabı yaz.\n"
            "Eğer kaynakta cevap açıkça varsa kaynak metindeki ifadeleri mümkün olduğunca koru.\n"
            "Diğer kaynaklardan gereksiz bilgi karıştırma.\n"
            "Cevap formatı kesinlikle şu olsun:\n"
            "Seçilen Kaynak: <kaynak id>\n"
            "Cevap: <cevap metni>"
        )

        user_prompt = f"""
[Kaynaklar]
{context_text}

[Soru]
{question}
""".strip()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def clean_generated_answer(self, raw: str) -> str:
        text = str(raw).strip()

        if "assistant" in text:
            text = text.split("assistant")[-1].strip()

        if "Cevap:" in text:
            text = text.split("Cevap:", 1)[1].strip()

        text = re.sub(
            r"^\s*Seçilen Kaynak\s*:.*?\n",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        text = re.sub(r"\n\s*Kaynak\s*:.*$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        text = re.sub(r"\n\s*ID\s*:.*$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

        return text.strip()

    def is_bad_answer(self, answer: str) -> bool:
        a = str(answer).strip()
        low = a.lower()

        if len(a) < 25:
            return True

        bad_starts = [
            "[id:",
            "[kaynak:",
            "kaynak:",
            "id:",
            "oricon_",
            "turkish_law_",
            "yargitay_",
            "train_kayit_",
            "kaynak 1",
            "kaynak 2",
        ]

        return any(low.startswith(x) for x in bad_starts)

    def _generate_raw(self, messages, max_new_tokens: int = None) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Generator is not loaded. Call load() first.")

        max_new_tokens = max_new_tokens or self.max_new_tokens

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        context_top_k: int = 5,
        retry: bool = True,
    ) -> Tuple[str, str, str, bool]:
        context_text = self.format_contexts(contexts, context_top_k=context_top_k)
        messages = self.build_messages(question, context_text)

        raw = self._generate_raw(messages)
        answer = self.clean_generated_answer(raw)

        used_retry = False

        if retry and self.is_bad_answer(answer):
            retry_system = (
                "Önceki cevap hatalıydı çünkü sadece kaynak/ID yazılmış olabilir.\n"
                "Şimdi yalnızca verilen kaynaklara dayanarak maddi cevabı açıkça yaz.\n"
                "Sadece ID yazma. Kaynak metindeki cevabı koruyarak cevap üret.\n"
                "Format:\n"
                "Cevap: <cevap metni>"
            )

            retry_messages = [
                {"role": "system", "content": retry_system},
                {"role": "user", "content": f"[Kaynaklar]\n{context_text}\n\n[Soru]\n{question}"},
            ]

            retry_raw = self._generate_raw(retry_messages)
            retry_answer = self.clean_generated_answer(retry_raw)

            raw = raw + "\n\n[RETRY_RAW]\n" + retry_raw
            answer = retry_answer
            used_retry = True

        return answer, raw, context_text, used_retry
