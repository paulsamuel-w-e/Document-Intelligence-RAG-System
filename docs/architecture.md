🏗️ System Architecture

Overview

The system follows a **multi-stage hybrid RAG architecture**:

Ingestion → Processing → Embedding → Storage → Hybrid Retrieval → Filtering → Generation → Evaluation

---

Layers

1. Ingestion Layer

* Extracts text via PyMuPDF
* OCR fallback via PaddleOCR
* Quality-based switching

---

2. Processing Layer

* Cleans noise (references, captions, page numbers)
* Splits into chunks
* Assigns metadata (section detection: abstract / intro / related / body)

---

3. Embedding Layer

* SentenceTransformer (MiniLM)
* Produces normalized dense vectors

---

4. Storage Layer

* FAISS (IndexFlatIP)
* Stores embeddings + metadata-rich chunks

---

5. Retrieval Layer (Hybrid)

* Dense search (semantic similarity)
* Sparse search (BM25 keyword matching)
* Merging (score aggregation)
* Metadata weighting (e.g., downweight “related work”)
* Optional reranking

---

6. Agent Layer

* Intent detection (summarize / extract / QA)
* Query-type classification (broad / fact / deep)
* Adaptive retrieval strategy (dynamic top_k)

---

7. Tools Layer

Task-specific prompt templates:

* Summarization → global synthesis
* Extraction → structured facts
* QA → grounded answers

(Note: Prompt control is currently basic and not strictly enforced)

---

8. LLM Layer

* OpenAI or local model
* Prompt-driven generation
* Output quality depends on model capability

---

9. Evaluation Layer

* Dataset-driven evaluation
* Retrieval vs Answer scoring
* Hallucination detection
* Difficulty-based analysis

---

Design Principles

* Modular and extensible
* Retrieval ≠ generation separation
* Evaluation-driven development
* Debuggable and observable pipeline
