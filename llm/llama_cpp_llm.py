from __future__ import annotations

from typing import Optional
from utils.logger import get_logger

from .llm_wrapper import BaseLLM

logger = get_logger(__name__)


class LlamaCppLLM(BaseLLM):
    """
    GGUF / llama.cpp based local LLM (e.g., Mistral).

    Uses llama-cpp-python backend.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 20,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> None:
        try:
            from llama_cpp import Llama  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Run: pip install llama-cpp-python") from exc

        logger.info("Loading GGUF model from %s...", model_path)

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,  # offload to GPU
            verbose=False,
        )

        self.temperature = temperature
        self.top_p = top_p

        logger.info("Llama.cpp model loaded.")

    def _format_prompt(self, prompt: str) -> str:
        """
        Apply Mistral prompt template.
        """
        return f"<s>[INST] {prompt} [/INST]"

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        formatted_prompt = self._format_prompt(prompt)

        output = self.llm(
            formatted_prompt,
            max_tokens=max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=["</s>"],
        )

        return output["choices"][0]["text"].strip()
    
    @property
    def backend_name(self) -> str:
        return "llama_cpp"