"""
Agent tools: stateless functions that wrap LLM calls for specific tasks.

Each tool:
- Receives retrieved document chunks
- Builds a structured prompt using prompt_builder
- Formats it based on the LLM backend
- Returns a generated response

This layer ensures:
- Consistent prompting across models (OpenAI, T5, Mistral)
- Strong grounding (context-only answering)
- Structured outputs for evaluation
"""

from llm.llm_wrapper import BaseLLM
from rag.prompt_builder import (
    build_context,
    qa_prompt,
    summarize_prompt,
    extract_prompt,
    format_for_model,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def summarize(chunks: list[str], llm: BaseLLM) -> str:
    """
    Generate a concise summary of retrieved document chunks.

    Uses:
        - Model-agnostic summarization prompt
        - Backend-specific formatting (e.g., Mistral INST)

    Args:
        chunks: List of retrieved text chunks
        llm:    LLM backend instance

    Returns:
        Structured summary string
    """
    context = build_context(chunks)
    base_prompt = summarize_prompt(context)
    final_prompt = format_for_model(base_prompt, llm.backend_name)

    logger.debug("Calling summarize tool.")
    return llm.generate(final_prompt, max_new_tokens=220)


def extract_key_info(chunks: list[str], llm: BaseLLM) -> str:
    """
    Extract structured key information (facts, entities, values)
    from retrieved chunks.

    Ensures:
        - No hallucinated facts
        - Bullet-point structured output

    Args:
        chunks: Retrieved document chunks
        llm:    LLM backend

    Returns:
        Bullet-point formatted string
    """
    context = build_context(chunks)
    base_prompt = extract_prompt(context)
    final_prompt = format_for_model(base_prompt, llm.backend_name)

    logger.debug("Calling extract_key_info tool.")
    return llm.generate(final_prompt, max_new_tokens=220)


def answer_question(query: str, chunks: list[str], llm: BaseLLM) -> str:
    """
    Answer a user question using retrieved document context.

    Features:
        - Strict grounding (context-only)
        - Structured output (Answer + Sources + Confidence)
        - Backend-aware prompt formatting

    Args:
        query:  User question
        chunks: Retrieved relevant chunks
        llm:    LLM backend

    Returns:
        Grounded answer string
    """
    context = build_context(chunks)
    base_prompt = qa_prompt(query, context)
    final_prompt = format_for_model(base_prompt, llm.backend_name)

    logger.debug("Calling answer_question tool.")
    return llm.generate(final_prompt, max_new_tokens=256)