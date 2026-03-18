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
        "Summarize the document based only on the context below.\n\n"
        "IMPORTANT:\n"
        "- Focus on the main topic, purpose, and key contributions\n"
        "- Combine information across chunks into a coherent summary\n"
        "- Do NOT copy phrases directly; paraphrase\n"
        "- If the information is not present, say so explicitly\n\n"
        "OUTPUT FORMAT:\n"
        "Summary: <3-5 sentence paragraph>\n"
        "SOURCES: [Chunk X], [Chunk Y]\n"
        "CONFIDENCE: low|medium|high\n\n"
        f"Context:\n{context}\n\n"
        "Write the required output now."
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
        "From the document context below, extract key facts, entities, dates, "
        "and important figures as bullet points. Include a source tag for each bullet.\n\n"
        "IMPORTANT:\n"
        "- Only include facts clearly supported by the context\n"
        "- Return bullets in the form: '- Fact (SOURCE: [Chunk N])'\n"
        "- If a fact is not present, do not invent it\n\n"
        f"Context:\n{context}\n\n"
        "Return only the bullet points."
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
        "- Combine information from multiple chunks when relevant\n"
        "- Provide both WHAT it is and WHY / how it works if asked\n"
        "- Cite the supporting chunk(s) explicitly at the end of the answer using chunk labels\n"
        "- If the answer is not present in the context, say: "
        "'The document does not contain enough information to answer this question.'\n"
        "- Output a one-line CONFIDENCE: low|medium|high\n\n"
        "OUTPUT FORMAT:\n"
        "Answer: <2-5 sentences>\n"
        "SOURCES: [Chunk X], [Chunk Y]\n"
        "CONFIDENCE: low|medium|high\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    
    logger.debug("Calling answer_question tool.")
    return llm.generate(prompt)