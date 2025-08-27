from __future__ import annotations
from typing import List, Dict
from .backends.base import ChatBackend, GenerationConfig

SYSTEM_TAG = "system"
USER_TAG = "user"
ASSISTANT_TAG = "assistant"

def format_messages(history: List[Dict[str, str]]) -> str:
    parts = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<{role}>{content}</{role}>")
    return "\n".join(parts) + f"\n<{ASSISTANT_TAG}>"

class ChatSession:
    def __init__(
            self,
            backend: ChatBackend,
            system_prompt: str = "You are a helpful assistant.",
            max_context_tokens: int | None = None,
            gen_config: GenerationConfig | None = None,
    ) -> None:
        self.backend = backend
        self.history: List[Dict[str, str]] = [
            {"role": SYSTEM_TAG, "content": system_prompt}
        ]
    
        self.max_content = min(
            backend.max_context(),
            max_context_tokens if max_context_tokens else backend.max_context(),
        )

        self.gen = gen_config or GenerationConfig(
            max_new_tokens = 200,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            stop=[f"\n<{USER_TAG}>", f"\n<{ASSISTANT_TAG}>"],
        )
    def _prompt_tokens(self) -> int:
        prompt = format_messages(self.history)
        return len(self.backend.tokenize(prompt))
    
    def _truncate_to_fit(self) -> None:
        while True:
            budget = self.max_content - self.gen.max_new_tokens
            if self._prompt_tokens() <= budget:
                break
            for i, msg in enumerate(self.history):
                if msg["role"] != SYSTEM_TAG:
                    del self.history[i]
                    break
            else:
                break

    def ask(self, user_text: str) -> str:
        self.history.append({"role": USER_TAG, "content": user_text})
        self._truncate_to_fit()
        prompt = format_messages(self.history)
        reply = self.backend.generate(prompt, self.gen)
        self.history.append({"role": ASSISTANT_TAG, "content": reply})
        return reply
