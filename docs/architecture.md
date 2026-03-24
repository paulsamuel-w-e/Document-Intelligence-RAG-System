# 🏗️ System Architecture

## Overview

The system follows a **multi-stage hybrid RAG architecture with reranking and controlled generation**:

Ingestion → Processing → Embedding → Storage → Hybrid Retrieval → Reranking → Context Construction → Generation → Evaluation

---

## Layers

### 1. Ingestion Layer

* PyMuPDF text extraction
* OCR fallback via PaddleOCR
* Quality-based switching

---

### 2. Processing Layer

* Noise cleaning (references, captions, page numbers)
* Chunking (fixed size + overlap)
* Section detection (abstract / intro / related / body)

---

### 3. Embedding Layer

* SentenceTransformer (MiniLM)
* Normalized dense embeddings

---

### 4. Storage Layer

* FAISS (IndexFlatIP)
* Stores embeddings + metadata-rich chunks

---

### 5. Retrieval Layer (Hybrid)

* Dense retrieval (semantic similarity)
* Sparse retrieval (BM25 keyword matching)
* Score merging (dense + sparse)
* Metadata weighting (section-aware scoring)

---

### 6. Reranking Layer

* Cross-encoder (MiniLM)
* Reorders top candidates based on query–chunk relevance

---

### 7. Agent Layer

* Intent detection (QA / summarize / extract)
* Query-type classification (broad / deep / fact)
* Adaptive retrieval (dynamic top_k)

---

### 8. Context Construction

* Concatenates top-ranked chunks
* Labels chunks ([Chunk X])
* Applies length constraints

⚠️ Current limitation: flat context (no hierarchy)

---

### 9. Tools Layer

Task-specific prompting:

* QA → grounded answers
* Summarization → concise synthesis
* Extraction → structured bullet points

---

### 10. LLM Layer

* OpenAI / HuggingFace / llama.cpp
* Prompt-driven generation
* Output quality depends on model capability

---

### 11. Evaluation Layer

* Keyword-based scoring
* Retrieval vs answer comparison
* Hallucination detection
* Difficulty-based analysis

---

## Design Principles

* Modular and extensible
* Retrieval ≠ generation separation
* Evaluation-driven development
* Debuggable and observable pipeline
