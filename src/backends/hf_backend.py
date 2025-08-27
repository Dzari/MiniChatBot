from __future__ import annotations

from typing import List, Optional, Sequence, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import (
    ChatBackend,
    GenerationConfig,
    BackendError,
    ConfigValidationError,
    truncate_at_first_stop, 
    )

class HFBackend(ChatBackend):
    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: str = "auto",
        dtype: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._device = self._pick_device(device)
        self._dtype = self._pick_dtype(dtype)

        try:
            self._tok = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=self._dtype if self._dtype is not None else None
            )
        except Exception as e:
            raise BackendError(f"Failed to load model/tokenizer '{model_name}': {e}") from e
        
        try: #type checker is misunderstanding that self._device is always a torch.device
            self._model.to(self._device) #type: ignore
            self._model.eval()
        except Exception as e:
            raise BackendError(f"Failed to place model on device '{self._device}': {e}") from e
        
        if self._tok.pad_token_id is None:
            if self._tok.eos_token_id is not None:
                self._tok.pad_token = self._tok.eos_token
            else:
                self._tok.add_special_tokens({"pad_token": "<|pad|>"})
                self._model.resize_token_embeddings(len(self._tok))

        self._max_ctx = self._infer_max_context()

    # -----------------------------
    # Contract methods
    # -----------------------------

    def tokenize(self, text: str) -> List[int]:
        try:
            return self._tok.encode(text, add_special_tokens=False)
        except Exception as e:
            raise BackendError(f"Tokenization failed: {e}") from e
        
    def detokenize(self, ids: Sequence[int]) -> str:
        try:
            return self._tok.decode(list(ids), skip_special_tokens=True)
        except Exception as e:
            raise BackendError(f"Detokenization failed: {e}") from e
    def max_context(self) -> int:
        return self._max_ctx
    
    @torch.inference_mode()
    def generate(self, prompt: str, gen: GenerationConfig) -> str:
        if not isinstance(gen, GenerationConfig):
            raise ConfigValidationError("gen must be a GenerationConfig instance")
        gen.validate()

        try:
            enc = self._tok(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(self._device)
            attn_mask = enc.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(self._device)
        except Exception as e:
            raise BackendError(f"Enconding prompt failed: {e}") from e
        
        prompt_len = input_ids.shape[1]
        if prompt_len + gen.max_new_tokens > self._max_ctx:
            raise BackendError(
                f"Prompt too long for context window: prompt={prompt_len}, "
                f"max_new={gen.max_new_tokens}, window={self._max_ctx}. "
                "Trim history or reduce max_new_tokens."
            )
        
        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(gen.max_new_tokens),
            "do_sample": True,
            "temperature": float(gen.temperature),
            "top_p": float(gen.top_p),
            "top_k": int(gen.top_k),
            "repetition_penalty": float(gen.repetition_penalty),
            "pad_token_id": self._tok.pad_token_id,
            "eos_token_id": self._tok.eos_token_id,
        }

        if attn_mask is not None:
            generate_kwargs["attention_mask"] = attn_mask
        
        try:
            out = self._model.generate(input_ids=input_ids, **generate_kwargs)
        except torch.cuda.OutOfMemoryError as e:
            raise BackendError("CUDA OOM during generation. Reduce max_new_tokens or batch size.")
        except Exception as e:
            raise BackendError(f"Generation failed: {e}") from e
        
        new_token_ids = out[0, prompt_len:]
        text = self._tok.decode(new_token_ids, skip_special_tokens=True)

        if gen.stop:
            text = truncate_at_first_stop(text, gen.stop)

        return text
    
    # -----------------------------
    # Internals
    # -----------------------------
    
    def _pick_device(self, requested: str) -> torch.device:
        r = (requested or "auto").lower()
        if r == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(r)
    
    def _pick_dtype(self, dtype_str: Optional[str]):
        if dtype_str is None:
            return None
        d = dtype_str.lower()
        if d in ("fp16", "float16"):
            return torch.float16
        if d in ("bf16", "bfloat16"):
            return torch.bfloat16
        if d in ("fp32", "float32"):
            return torch.float32
        raise BackendError(f"Unsupported dtype string: {dtype_str}")
    
    def _infer_max_context(self) -> int:
        cfg = getattr(self._model, "config", None)

        for field in ("max_postion_embeddings", "n_ctx", "n_positions"):
            val = getattr(cfg, field, None)
            if isinstance(val, int) and val > 0:
                return val
        try:
            v = int(getattr(self._tok, "model_max_length", 1024))
            if v > 0 and v < 10**9:
                return v
        except Exception:
            pass
        return 1024
    
    def __repr__(self) -> str:
        return f"HFBackend(model='{self._model_name}', device='{self._device}', ctx={self._max_ctx})"