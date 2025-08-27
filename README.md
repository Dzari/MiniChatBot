# Mini‑ChatGPT (Python + PyTorch)

A **model‑agnostic chat system** with a swappable backend. Currently this project just wraps a small pretrained Hugging Face language model for a working chat UI. Eventually I plan to swap in a tiny Transformer I build/train myself—without changing the app surface.

---

## What this is
- A compact chatbot that separates **UI / chat loop / model backend** via a small interface (`ChatBackend`).
- Designed to demonstrate **clean Python architecture** and **ML inference fundamentals**, not SOTA accuracy.
- Shows how **tokenization, context windows, and decoding** shape behavior.

## Skills demonstrated
- **Python engineering:** `@dataclass`, `@abstractmethod`, custom exceptions, typed interfaces, package structure, relative/absolute imports.
- **ML systems:** prompt formatting for chat, **token budgeting** (prompt + `max_new_tokens` ≤ context), device/dtype handling (CPU/CUDA/MPS; fp32/fp16/bf16), error paths (OOM/validation).
- **Decoding controls:** temperature, top‑p, top‑k, repetition penalty, **stop sequences** and post‑processing.
- **Product/UI:** minimal **Gradio** chat interface; backend factory for runtime model selection.

## Tech used
- **PyTorch** (tensor runtime)  
- **Hugging Face Transformers** (tokenizer + causal LM, Track A)  
- **Gradio** (lightweight chat UI)  
- *(Planned)* **FastAPI** (HTTP API), **PEFT/LoRA** (instruction‑tuning), **FAISS** (RAG)

## Architecture (at a glance)
```
UI (Gradio)
  → Chat loop (format history, truncate to fit, call generate)
  → Backend interface (`ChatBackend` + `GenerationConfig`)
       ├─ HFBackend  (Transformers; current)
       └─ TinyBackend (custom TinyGPT; planned)
```

## Key ML concepts practiced
- Tokenization vs characters; **context windows** and truncation strategy.  
- **Sampling**: temperature (randomness), top‑p/top‑k (candidate pool), repetition penalty (anti‑loops).  
- Deterministic **stop‑token** cutting for clean turn boundaries.

## Roadmap
- Config via YAML/ENV; API server (FastAPI).  
- Unit tests: backend contract, truncation guarantees, stop‑handling.  
- Track B: implement/train **TinyGPT**, plug in as `TinyBackend`.  
- Stretch: RAG with FAISS; tool/function calling; small LoRA finetunes.

## Reviewer checklist
- Clear interface boundaries; UI depends only on `ChatBackend`.  
- Correct token budgeting and prompt slicing; stop sequences respected.  
- Sensible decoding defaults; device/dtype switching works.  
- Swapping to an **instruction‑tuned** model yields markedly better coherence.
