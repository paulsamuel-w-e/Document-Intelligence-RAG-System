"""
LLM abstraction layer.
Supports:
  - OpenAI (GPT-4.1-mini via API)
  - Local HuggingFace model (flan-t5-base) using AutoModelForSeq2SeqLM

Both expose the same interface: generate(prompt) -> str
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

import os


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseLLM(ABC):
    """Common interface for all LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate a response for the given prompt."""
        ...


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class OpenAILLM(BaseLLM):
    """
    Thin wrapper around the OpenAI chat completions API.

    Args:
        model:   OpenAI model name (default: gpt-4.1-mini).
        api_key: API key. If None, reads from OPENAI_API_KEY env var.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        api_key: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Run: pip install openai") from exc

        self.model = model

        # Read from env if not passed
        api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self._client = OpenAI(api_key=api_key)

        logger.info("OpenAI LLM initialised (model=%s).", model)

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Local HuggingFace backend (flan-t5-base)
# ---------------------------------------------------------------------------

class LocalLLM(BaseLLM):
    """
    Seq2Seq model wrapper using AutoModelForSeq2SeqLM + tokenizer.
    Does NOT use the HuggingFace pipeline abstraction.

    Args:
        model_name: HuggingFace model ID (default: google/flan-t5-base).
        device:     "cpu", "cuda", or "mps". Auto-detected if None.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: Optional[str] = None,
    ) -> None:
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Run: pip install torch transformers") from exc

        import torch  # noqa: PLC0415, F811
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # noqa: PLC0415, F811

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name

        logger.info("Loading local LLM '%s' on %s...", model_name, device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self._model.eval()
        logger.info("Local LLM loaded.")

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        import torch  # noqa: PLC0415

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
            )

        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_llm(backend: str = "openai", **kwargs) -> BaseLLM:
    """
    Factory to instantiate the correct LLM backend.

    Args:
        backend: "openai" or "local".
        **kwargs: Forwarded to the backend constructor.

    Returns:
        An initialised BaseLLM instance.
    """
    backend = backend.lower()
    if backend == "openai":
        return OpenAILLM(**kwargs)
    elif backend == "local":
        return LocalLLM(**kwargs)
    else:
        raise ValueError(f"Unknown LLM backend: '{backend}'. Choose 'openai' or 'local'.")