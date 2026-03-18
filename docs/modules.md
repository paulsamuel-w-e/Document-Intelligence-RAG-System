🧩 Module Breakdown

ingestion/

* PDF loader
* OCR fallback
* Quality heuristics

---

rag/embeddings.py

* SentenceTransformer wrapper
* Batch encoding

---

rag/vectorstore.py

* FAISS storage
* Stores metadata-rich chunks

---

rag/bm25.py (NEW)

* Sparse retrieval using BM25

---

rag/retriever.py

* Hybrid retrieval (dense + sparse)
* Score merging
* Metadata weighting
* Reranking support
* Dual interface:

  * retrieve() → production
  * retrieve_with_metadata() → evaluation

---

rag/splitter.py

* Text cleaning
* Chunking
* Section detection

---

agents/agent.py

* Intent detection
* Query-type classification
* Adaptive retrieval
* Tool routing

---

agents/tools.py

* Summarization
* Extraction
* QA
* Structured prompting

---

llm/llm_wrapper.py

* BaseLLM interface
* OpenAI backend
* Local HuggingFace backend

---

eval/ (NEW)

* eval_data.json (dataset)
* evaluator.py (metrics + diagnostics)

---

utils/logger.py

* Central logging system
