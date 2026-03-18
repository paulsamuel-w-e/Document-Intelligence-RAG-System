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
    combined = "\n\n".join(
        f"[Chunk {i+1}]\n{chunk}"
        for i, chunk in enumerate(chunks)
    )
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
        "You are a document summarization assistant.\n\n"
        "Summarize the document based on the context below.\n\n"
        "IMPORTANT:\n"
        "- Focus on the main topic, purpose, and key contributions\n"
        "- Combine information across chunks into a coherent summary\n"
        "- Do NOT copy phrases directly\n"
        "- Ignore irrelevant or repeated details\n\n"
        f"Context:\n{context}\n\n"
        "Write a clear summary in 3-5 sentences."
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
        "Extract key facts from the document below.\n\n"
        "IMPORTANT:\n"
        "- Only include clearly supported facts\n"
        "- Avoid duplicates\n"
        "- Group related information if possible\n\n"
        f"Context:\n{context}\n\n"
        "Return bullet points."
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
        "You are a precise and technical question-answering assistant.\n\n"
        "Use ONLY the context below to answer the question.\n\n"
        "IMPORTANT:\n"
        "- Combine information from multiple chunks\n"
        "- Explain the concept clearly, not just a phrase\n"
        "- Include both WHAT it is and WHY it is used\n"
        "- Do NOT copy sentences directly\n"
        "- Do NOT include unrelated information\n\n"
        "FORMAT:\n"
        "Answer in 2–5 sentences with a clear explanation.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    logger.debug("Calling answer_question tool.")
    return llm.generate(prompt)