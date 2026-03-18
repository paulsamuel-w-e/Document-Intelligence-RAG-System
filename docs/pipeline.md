🔄 End-to-End Pipeline

1. Document Loading
   Input: PDF
   Output: raw text

---

2. Text Extraction

* PyMuPDF primary extraction
* OCR fallback if quality is low

---

3. Text Cleaning
   Removes:

* references
* captions
* page numbers

---

4. Chunking

* Fixed-size chunking (500 / overlap 100)
* Noise filtering
* Section tagging (abstract, intro, related, body)

---

5. Embedding Generation

* Dense vector encoding (MiniLM)
* Normalized for cosine similarity

---

6. Vector Storage

* FAISS index
* Stores embeddings + metadata

---

7. Query Processing

a. Query Understanding

* Intent detection
* Query type classification

b. Hybrid Retrieval

* Dense retrieval (semantic)
* Sparse retrieval (BM25)
* Merge + score aggregation

c. Filtering & Ranking

* Metadata weighting
* Optional reranking

d. Context Construction

* Top-k chunk selection
* Flattened context (current limitation)

e. Tool Execution

* QA / summarize / extract

f. LLM Generation

* Prompt-driven output

---

8. Evaluation (NEW)

* Retrieval scoring
* Answer scoring
* Hallucination detection
* Difficulty-based reporting

---

Output

* Answer / Summary / Extracted facts
* Evaluation metrics (optional mode)
