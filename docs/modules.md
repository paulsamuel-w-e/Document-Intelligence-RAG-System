# 🧩 Module Breakdown

## ingestion/

* PDF loader
* OCR fallback

---

## rag/embeddings.py

* SentenceTransformer wrapper
* Batch encoding

---

## rag/vectorstore.py

* FAISS-based storage
* Add/search/save/load

---

## rag/retriever.py

* Query encoding
* Top-k retrieval

---

## rag/splitter.py

* Text cleaning
* Chunking logic

---

## agents/agent.py

* Intent detection
* Pipeline orchestration

---

## agents/tools.py

* QA
* Summarization
* Extraction

---

## llm/llm_wrapper.py

* BaseLLM interface
* OpenAI backend
* Local HF backend

---

## utils/logger.py

* Central logging system
