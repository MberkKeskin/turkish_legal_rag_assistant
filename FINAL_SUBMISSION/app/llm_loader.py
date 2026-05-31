
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_qwen_lora_model(
    lora_dir: str,
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    load_in_4bit: bool = True,
):
    """
    Loads base Qwen2.5-3B-Instruct and attaches a LoRA adapter.
    """

    clear_gpu()

    tokenizer = AutoTokenizer.from_pretrained(
        lora_dir,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    model = PeftModel.from_pretrained(
        base_model,
        lora_dir,
    )

    model.eval()

    return model, tokenizer
