"""
Agent tools: stateless functions that wrap LLM calls for specific tasks.
Each tool receives a list of chunks and an LLM, returns a string result.
"""

from llm.llm_wrapper import BaseLLM
from utils.logger import get_logger

logger = get_logger(__name__)

_MAX_CONTEXT_CHARS = 3000  # guard against token overflow


def _truncate_context(chunks: list[str]) -> str:
    """Join chunks and truncate to a safe length."""
    combined = "\n\n".join(chunks)
    if len(combined) > _MAX_CONTEXT_CHARS:
        combined = combined[:_MAX_CONTEXT_CHARS] + "\n...[truncated]"
    return combined


def summarize(chunks: list[str], llm: BaseLLM) -> str:
    """
    Produce a concise summary of the provided document chunks.

    Args:
        chunks: Relevant text chunks to summarise.
        llm:    LLM backend to use for generation.

    Returns:
        A paragraph-length summary string.
    """
    context = _truncate_context(chunks)
    prompt = (
        "You are a document summarisation assistant.\n\n"
        "Below is extracted content from a document:\n\n"
        f"{context}\n\n"
        "Write a clear, concise summary (3-5 sentences) covering the main points. "
        "Do not add information that is not present in the text."
    )
    logger.debug("Calling summarize tool.")
    return llm.generate(prompt)


def extract_key_info(chunks: list[str], llm: BaseLLM) -> str:
    """
    Extract structured key information (entities, dates, facts) from chunks.

    Args:
        chunks: Relevant text chunks.
        llm:    LLM backend to use.

    Returns:
        Bullet-point list of key facts.
    """
    context = _truncate_context(chunks)
    prompt = (
        "You are an information extraction assistant.\n\n"
        "From the document content below, extract key facts, entities, dates, "
        "and important figures as a bullet-point list:\n\n"
        f"{context}\n\n"
        "Return only the bullet points. Do not include unsupported claims."
    )
    logger.debug("Calling extract_key_info tool.")
    return llm.generate(prompt)


def answer_question(query: str, chunks: list[str], llm: BaseLLM) -> str:
    """
    Answer a specific user question grounded in the retrieved chunks.

    Args:
        query:  The user's natural language question.
        chunks: Retrieved context chunks.
        llm:    LLM backend.

    Returns:
        A grounded answer string.
    """
    context = _truncate_context(chunks)
    prompt = (
        "You are a precise question-answering assistant.\n\n"
        "Use ONLY the context below to answer the question. "
        "If the answer is not present in the context, say: "
        "'The document does not contain enough information to answer this question.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    logger.debug("Calling answer_question tool.")
    return llm.generate(prompt)