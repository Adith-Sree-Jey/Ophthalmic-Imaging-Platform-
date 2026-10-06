"""
medgemma_engine.py — MedGemma 4B with 4-bit quantization for RTX 2060 (6GB VRAM).

4-bit quantization loads MedGemma in ~3.5GB VRAM — fits alongside V7 model.
Generation drops from ~5 minutes on CPU to ~5-10 seconds on GPU.

Requirements:
    pip install transformers accelerate bitsandbytes

Hugging Face model:
    google/medgemma-4b-it  (instruction-tuned, text-only)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── Config ────────────────────────────────────────────────────────────────────

def _env_value(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip()
    if not value or value.startswith("REPLACE_ME"):
        return default
    return value


def _env_int(name: str, default: int) -> int:
    value = _env_value(name, str(default))
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = _env_value(name, str(default))
    try:
        return float(value)
    except ValueError:
        return default


HF_MODEL_ID = _env_value("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it")
HF_TOKEN = _env_value("HF_TOKEN", "")
MAX_NEW_TOKENS = _env_int("MEDGEMMA_MAX_TOKENS", 512)
TEMPERATURE = _env_float("MEDGEMMA_TEMPERATURE", 0.2)  # lower = faster + consistent
WEB_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_MEDGEMMA = WEB_DIR / "backend" / "models" / "medgemma-4b-it"
_configured_local_path = Path(
    _env_value("LOCAL_MEDGEMMA_PATH", str(DEFAULT_LOCAL_MEDGEMMA))
).expanduser()
LOCAL_MEDGEMMA = str(
    _configured_local_path
    if _configured_local_path.is_absolute()
    else WEB_DIR / _configured_local_path
)

# ── Singleton ─────────────────────────────────────────────────────────────────

_tokenizer = None
_model     = None
_device    = None


def _load_model():
    global _tokenizer, _model, _device

    if _model is not None:
        return

    model_source = LOCAL_MEDGEMMA if os.path.exists(LOCAL_MEDGEMMA) else HF_MODEL_ID
    print(f"[MedGemma] Loading {model_source} with 4-bit quantization …")
    t0 = time.time()

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    _tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        token=HF_TOKEN,
        local_files_only=model_source == LOCAL_MEDGEMMA,
    )

    if _device == "cuda":
        # 4-bit quantization — fits MedGemma 4B in ~3.5GB VRAM
        # Works on RTX 2060 (6GB) alongside V7 model (~3GB)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",             # NormalFloat4 — best quality
            bnb_4bit_compute_dtype=torch.float16,  # compute in fp16 for speed
            bnb_4bit_use_double_quant=True,        # nested quantization saves ~0.4GB
        )
        _model = AutoModelForCausalLM.from_pretrained(
            model_source,
            token=HF_TOKEN,
            quantization_config=bnb_config,
            device_map="auto",
            local_files_only=model_source == LOCAL_MEDGEMMA,
        )
        print(f"[MedGemma] Ready on GPU (4-bit) ({time.time() - t0:.1f}s)")
        print(f"[MedGemma] VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

    else:
        # CPU fallback if CUDA unavailable
        _model = AutoModelForCausalLM.from_pretrained(
            model_source,
            token=HF_TOKEN,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            local_files_only=model_source == LOCAL_MEDGEMMA,
        )
        print(f"[MedGemma] Ready on CPU ({time.time() - t0:.1f}s)")
        print("[MedGemma] Warning: CPU mode is slow (~5 min per report)")

    _model.eval()


def generate(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """
    Run a single MedGemma generation pass.

    Parameters
    ----------
    prompt : str
        Instruction prompt string.
    max_new_tokens : int
        Max tokens to generate (default 512 — enough for fuller report text).

    Returns
    -------
    str
        Generated text, stripped.
    """
    _load_model()

    messages = [{"role": "user", "content": prompt}]

    # apply_chat_template returns tensor or BatchEncoding — handle both
    raw = _tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    if isinstance(raw, torch.Tensor):
        input_ids = raw.to(_device)
    elif hasattr(raw, "input_ids"):
        input_ids = raw.input_ids.to(_device)
    else:
        input_ids = raw["input_ids"].to(_device)

    prompt_len     = input_ids.shape[-1]
    attention_mask = torch.ones_like(input_ids).to(_device)

    try:
        with torch.inference_mode():
            output_ids = _model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=_tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][prompt_len:]
        result  = _tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        print(f"[MedGemma] Generated {len(new_ids)} tokens")
        return result

    except Exception as exc:
        import traceback
        print(f"[MedGemma] Generation failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise


def unload():
    """Free GPU memory."""
    global _tokenizer, _model
    if _model is not None:
        del _model
        _model = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[MedGemma] Model unloaded.")
