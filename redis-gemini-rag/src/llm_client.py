from __future__ import annotations

import hashlib

import numpy as np
import requests
from google import genai
from google.genai import types


class LLMClient:
    def __init__(
        self,
        api_key: str,
        embedding_model: str,
        chat_model: str,
        embedding_dim: int,
        llm_provider: str = "gemini",
        embedding_provider: str = "gemini",
        deepseek_api_key: str = "",
        deepseek_base_url: str = "https://api.deepseek.com",
        deepseek_chat_model: str = "",
        debug: bool = False,
    ) -> None:
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.embedding_dim = embedding_dim
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.deepseek_api_key = deepseek_api_key
        self.deepseek_base_url = deepseek_base_url.rstrip("/")
        self.deepseek_chat_model = deepseek_chat_model
        self.debug = debug
        self.client = genai.Client(api_key=api_key) if api_key else None

    def _error_message(self, exc: Exception, provider: str) -> str:
        base = f"{provider} API request failed. Please check your API key and internet connection."
        if not self.debug:
            return base
        details = str(exc).replace("\n", " ").strip()
        if len(details) > 300:
            details = f"{details[:300]}..."
        return f"{base} Debug: {exc.__class__.__name__}: {details}"

    def get_embedding(self, text: str, task_type: str = "retrieval_document") -> np.ndarray:
        if self.embedding_provider == "local":
            return self._get_local_embedding(text)
        if self.client is None:
            raise RuntimeError("Gemini client is not configured for embeddings.")
        try:
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=task_type.upper(),
                    output_dimensionality=self.embedding_dim,
                ),
            )
            embedding = result.embeddings[0].values
            return np.array(embedding, dtype=np.float32)
        except Exception as exc:  # pragma: no cover - network/API dependent
            raise RuntimeError(self._error_message(exc, "Gemini")) from exc

    def _get_local_embedding(self, text: str) -> np.ndarray:
        vector = np.zeros(self.embedding_dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vector
        for word in words:
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest, byteorder="big") % self.embedding_dim
            vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def _ask_gemini(self, prompt: str) -> str:
        if self.client is None:
            raise RuntimeError("Gemini client is not configured for chat.")
        try:
            response = self.client.models.generate_content(
                model=self.chat_model,
                contents=prompt,
            )
            return (response.text or "").strip()
        except Exception as exc:  # pragma: no cover - network/API dependent
            raise RuntimeError(self._error_message(exc, "Gemini")) from exc

    def _ask_deepseek(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.deepseek_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.deepseek_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.deepseek_chat_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
        except Exception as exc:  # pragma: no cover - network/API dependent
            raise RuntimeError(self._error_message(exc, "DeepSeek")) from exc

    def ask_llm(self, prompt: str) -> str:
        if self.llm_provider == "deepseek":
            return self._ask_deepseek(prompt)
        return self._ask_gemini(prompt)
