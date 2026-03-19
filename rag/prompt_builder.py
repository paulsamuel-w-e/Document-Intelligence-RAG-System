# rag/prompt_builder.py

from typing import List


def build_context(chunks: list[str], max_chars: int = 3000) -> str:
    """
    Build a formatted context string from retrieved chunks.

    - Adds chunk labels
    - Enforces max length to prevent token overflow
    """

    context = "\n\n".join(
        f"[Chunk {i+1}]\n{chunk}"
        for i, chunk in enumerate(chunks)
    )

    if len(context) > max_chars:
        context = context[:max_chars] + "\n...[truncated]"

    return context


# -------------------------------------------------------------------
# TASK PROMPTS (MODEL-AGNOSTIC)
# -------------------------------------------------------------------

def qa_prompt(query: str, context: str) -> str:
    return f"""
You are a strict document question-answering system.

Your job is to answer using ONLY the provided context.

Rules:
- Do NOT use external knowledge
- Do NOT guess or assume
- Use exact terms from the context where possible
- Keep answers concise and clear

SPECIAL CASES:

1. If the question asks about something absent or not used:
   → Clearly state that it is NOT mentioned or NOT used in the context

2. If the question asks "why", "how", or "explain":
   → Provide a structured explanation using only context-supported details

3. If the answer is truly not present:
   → Respond exactly: Not found

Context:
{context}

Question:
{query}

Answer:
""".strip()


def summarize_prompt(context: str) -> str:
    return f"""
Summarize the document using only the context.

Focus on:
- main topic
- purpose
- key points

Rules:
- 3–5 sentences
- No repetition
- No assumptions

Context:
{context}

Summary:
""".strip()


def extract_prompt(context: str) -> str:
    return f"""
Extract key information from the context.

Rules:
- Bullet points only
- One fact per bullet
- Only include information explicitly present
- No explanations or assumptions

Context:
{context}

Output:
""".strip()


# -------------------------------------------------------------------
# MODEL-SPECIFIC WRAPPING
# -------------------------------------------------------------------

def format_for_model(prompt: str, backend: str) -> str:
    backend = backend.lower()

    if backend == "llama_cpp":
        return f"[INST] {prompt} [/INST]"

    elif backend == "openai":
        return prompt  # handled separately in API

    elif backend == "local":
        return prompt  # T5 (no special format)

    return prompt