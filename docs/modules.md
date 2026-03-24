# 🧩 Module Breakdown

## ingestion/

* PDF loader (PyMuPDF)
* OCR fallback (PaddleOCR)
* Quality heuristics for switching

---

## rag/embeddings.py

* SentenceTransformer wrapper (MiniLM)
* Batch encoding
* Normalized embeddings (cosine-friendly)

---

## rag/vectorstore.py

* FAISS (IndexFlatIP)
* Stores embeddings + metadata-rich chunks
* Supports save/load persistence

---

## rag/bm25.py

* Sparse retrieval using BM25
* Token-based keyword matching

---

## rag/retriever.py

Hybrid retrieval engine:

* Dense retrieval (semantic)
* Sparse retrieval (BM25)
* Score merging (dense + sparse)
* Metadata weighting (section-aware scoring)
* Cross-encoder reranking (optional)

Interfaces:

* `retrieve()` → production use
* `retrieve_with_metadata()` → evaluation/debugging

---

## rag/reranker.py

* Cross-encoder (MiniLM)
* Re-scores query–chunk pairs
* Improves final relevance ordering

---

## rag/splitter.py

* Text cleaning (remove noise, references, captions)
* Chunking (RecursiveCharacterTextSplitter)
* Section detection:

  * abstract
  * introduction
  * related work
  * body
* Noise filtering (length + alpha ratio)

---

## rag/prompt_builder.py

* Context construction (chunk labeling + truncation)
* Task-specific prompts:

  * QA (strict grounding + negation handling)
  * Summarization (concise synthesis)
  * Extraction (structured bullet points)
* Backend-aware formatting (OpenAI / llama.cpp / local)

---

## agents/agent.py

* Intent detection (summarize / extract / QA)
* Query-type classification:

  * broad
  * deep
  * fact
* Adaptive retrieval (dynamic `top_k`)
* Tool routing

---

## agents/tools.py

Task-specific execution layer:

* `summarize()` → document-level summary
* `extract_key_info()` → structured fact extraction
* `answer_question()` → grounded QA

Responsibilities:

* Build context
* Apply prompts
* Call LLM backend

---

## llm/llm_wrapper.py

* BaseLLM interface
* OpenAI backend (chat API)
* Local HuggingFace backend (Flan-T5)
* Factory (`get_llm`)

---

## llm/llama_cpp_llm.py

* llama.cpp backend support
* GGUF model inference (e.g., Mistral)
* Local high-performance inference option

---

## eval/

### evaluator.py

* Keyword-based answer scoring (with fuzzy matching)
* Retrieval scoring (keyword presence in chunks)
* Hallucination detection (context grounding)
* Answer depth scoring
* Negation-aware penalties
* Diagnostic outputs (missing keywords, evidence)

### eval_data.json

* Multi-difficulty evaluation dataset:

  * easy
  * medium
  * hard
  * tricky

---

## utils/logger.py

* Centralized logging system
* Consistent formatting across modules

---

## test/

### test_rag.py

* End-to-end pipeline execution
* Builds system from PDF
* Runs queries through agent

### test_eval.py

* Evaluation pipeline runner
* Computes metrics across dataset
* Reports performance + latency
