# 🔄 End-to-End Pipeline

## 1. Document Loading

Input: PDF
Output: raw text

---

## 2. Text Extraction

* PyMuPDF primary extraction
* OCR fallback (PaddleOCR) if quality is low

---

## 3. Text Cleaning

Removes:

* references
* captions
* page numbers
* excessive whitespace

---

## 4. Chunking & Metadata

* Fixed-size chunking (500, overlap 100)
* Noise filtering
* Section tagging:

  * abstract
  * introduction
  * related work
  * body

---

## 5. Embedding Generation

* SentenceTransformer (MiniLM)
* Produces normalized dense vectors

---

## 6. Vector Storage

* FAISS index (IndexFlatIP)
* Stores embeddings + metadata-rich chunks

---

## 7. Query Processing

### a. Query Understanding

* Intent detection (QA / summarize / extract)
* Query-type classification:

  * broad → document-level
  * deep → reasoning/explanation
  * fact → direct lookup

---

### b. Hybrid Retrieval

* Dense retrieval (semantic similarity)
* Sparse retrieval (BM25 keyword matching)
* Merge results (score aggregation)

---

### c. Metadata Weighting

* Downweight low-value sections (e.g., related work)

---

### d. Reranking

* Cross-encoder reorders top candidates
* Improves final relevance

---

### e. Context Construction

* Top-k chunk selection
* Chunk labeling: `[Chunk X]`
* Length control (truncate if needed)

⚠️ Current limitation: flat context (no hierarchy)

---

### f. Tool Execution

Based on intent:

* QA → grounded answer generation
* Summarization → document synthesis
* Extraction → structured facts

---

### g. LLM Generation

* Prompt-driven output
* Strict grounding enforced
* Handles negation and missing information

---

## 8. Evaluation

* Retrieval scoring (keyword coverage in chunks)
* Answer scoring (keyword + depth)
* Hallucination detection (context mismatch)
* Difficulty-based reporting

---

## Output

* Answer / Summary / Extracted facts
* Evaluation metrics (optional mode)
* Latency tracking
