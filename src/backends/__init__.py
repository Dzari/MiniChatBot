from typing import Optional
from .base import ChatBackend
from .hf_backend import HFBackend

def create_backend(
        backend:str = "hf",
        model_name: str = "distilgpt2",
        device: str = "auto",
        dtype: Optional[str] = None,
) -> ChatBackend:
    b = backend.lower()
    if b == "hf":
        return HFBackend(model_name=model_name, device=device, dtype=dtype)
    raise ValueError(f"Unknown Backend: {backend}")