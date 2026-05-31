from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


class Generator:
    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        local_only: bool = False,
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.model_path = model_path or ""

        if self.model_path and Path(self.model_path).exists():
            source = self.model_path
            local_only = True
        else:
            source = model_name

        self.source = source
        config = AutoConfig.from_pretrained(source, local_files_only=local_only)
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_only)

        if config.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(source, local_files_only=local_only)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(source, local_files_only=local_only)

            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if getattr(self.model.config, "pad_token_id", None) is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.to(self.device)

    def _safe_context_limit(self, max_new_tokens: int) -> int | None:
        candidates = []

        if getattr(self.model.config, "max_position_embeddings", None):
            candidates.append(int(self.model.config.max_position_embeddings))

        if getattr(self.model.config, "n_positions", None):
            candidates.append(int(self.model.config.n_positions))

        tok_max = getattr(self.tokenizer, "model_max_length", None)
        if tok_max is not None:
            try:
                tok_max = int(tok_max)
                # tokenizer bazen absürt placeholder değer verir, onları ignore et
                if 0 < tok_max < 100000:
                    candidates.append(tok_max)
            except Exception:
                pass

        if not candidates:
            return 2048 - int(max_new_tokens)

        max_context = min(candidates)

        # çok küçükse de saçma olmasın
        max_context = max(512, max_context)

        return max(1, max_context - int(max_new_tokens))

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        max_input_tokens = self._safe_context_limit(max_new_tokens)

        tokenizer_kwargs = {
            "return_tensors": "pt",
            "truncation": True,
        }

        if max_input_tokens is not None:
            tokenizer_kwargs["max_length"] = max_input_tokens

        original_truncation_side = getattr(self.tokenizer, "truncation_side", "right")
        self.tokenizer.truncation_side = "left"
        inputs = self.tokenizer(prompt, **tokenizer_kwargs)
        self.tokenizer.truncation_side = original_truncation_side

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
