"""
Minimal, stable contract for all text-generation backends.

- GenerationConfig: parameters for one generation call.
- ChatBackend: abstract interface any backend must implement (HF, TinyGPT, etc.).

Keep this file import-light and dependency-free so tests and tooling can import it
without pulling in heavy ML libraries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence


# -----------------------------
# Exceptions
# -----------------------------
class ConfigValidationError(ValueError):
    """Raised when a GenerationConfig contains invalid values."""

class BackendError(RuntimeError):
    """Raised by concrete backends for operational errors (e.g., OOM, not loaded)."""

# -----------------------------
# Data object: GenerationConfig
# -----------------------------
@dataclass(slots=True)
class GenerationConfig:
    """
    Serializable decoding parameters for a single generation call.

    Notes:
    - 'stop' is a list of *strings*; backends should cut output at the first
      occurrence of any stop string (after decoding).
    - Top-k/top-p are optional; either (or both) may be used. Setting top_k=0
      can be interpreted by backends as "no top-k".
    """

    max_new_tokens: int = 200
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate field ranges and types; raise ConfigValidationError on issues."""
        errors: List[str] = []

        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens <= 0:
            errors.append("max_new_tokens must be a positive integer")

        if not isinstance(self.temperature, (int, float)) or self.temperature <= 0:
            errors.append("temperature must be a positive number")

        if not isinstance(self.top_p, (int, float)) or not (0.0 <= float(self.top_p) <= 1.0):
            errors.append("top_p must be in the range [0.0, 1.0]")

        if not isinstance(self.top_k, int) or self.top_k < 0:
            errors.append("top_k must be an integer >= 0")

        if not isinstance(self.repetition_penalty, (int, float)) or self.repetition_penalty < 1.0:
            errors.append("repetition_penalty must be >= 1.0")

        if not isinstance(self.stop, list) or any(not isinstance(s, str) for s in self.stop):
            errors.append("stop must be a list of strings")

        # Optional: normalize stop list (dedupe while preserving order, drop empties)
        if isinstance(self.stop, list):
            seen = set()
            normalized = []
            for s in self.stop:
                if s and s not in seen:
                    normalized.append(s)
                    seen.add(s)
            self.stop = normalized

        if errors:
            raise ConfigValidationError("; ".join(errors))

    def with_updates(self, **kwargs) -> "GenerationConfig":
        """
        Return a shallow-copied config with given fields updated and validated.
        Useful for per-request tweaks without mutating a shared instance.
        """
        params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "stop": list(self.stop),  # copy
        }
        params.update(kwargs)
        new_cfg = GenerationConfig(**params)
        return new_cfg

    def as_dict(self) -> dict:
        """Serialize to a plain dict (e.g., for JSON or logging)."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "stop": list(self.stop),
        }


# -----------------------------
# Interface: ChatBackend
# -----------------------------
class ChatBackend(ABC):
    """
    Minimal interface your app depends on for text generation.

    Implementations MUST:
      - be safe to call from a typical single-process UI loop,
      - respect GenerationConfig limits (especially max_new_tokens),
      - return ONLY the newly generated continuation from `generate(...)`,
      - cut the decoded continuation at the first occurrence of any stop string,
      - expose a stable max_context() in *tokens*.
    """

    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """
        Convert text to token IDs using the backend's tokenizer.

        Deterministic; no side effects.
        """
        raise NotImplementedError

    @abstractmethod
    def detokenize(self, ids: Sequence[int]) -> str:
        """
        Convert token IDs back to text.

        Should approximate the inverse of tokenize on valid sequences.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, gen: GenerationConfig) -> str:
        """
        Generate a continuation for `prompt` according to `gen`.

        Returns:
            The *decoded* continuation string (NOT including the prompt).
            If any stop string appears, output is truncated at its first occurrence.

        Errors:
            Should raise BackendError (or a subclass) for operational failures
            (e.g., model not loaded, device unavailable, OOM), and
            ConfigValidationError for invalid `gen`.
        """
        raise NotImplementedError

    @abstractmethod
    def max_context(self) -> int:
        """
        Maximum safe input length in tokens for this backend/model instance.

        Callers should use this to budget prompt tokens + max_new_tokens.
        """
        raise NotImplementedError


def truncate_at_first_stop(text: str, stops: Iterable[str]) -> str:
    """
    Return `text` truncated at the earliest first occurrence of any stop string.
    If none found, return `text` unchanged.
    """
    earliest = None
    for s in stops or []:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1:
            earliest = idx if earliest is None else min(earliest, idx)
    return text if earliest is None else text[:earliest]
