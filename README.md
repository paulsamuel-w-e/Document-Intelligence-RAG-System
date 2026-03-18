# 📄 Document Intelligence RAG System

A modular **Retrieval-Augmented Generation (RAG)** pipeline for document understanding, supporting:

* Question Answering
* Summarization
* Key Information Extraction

Built with a clean, extensible architecture using FAISS, sentence-transformers, and pluggable LLM backends.

---

## 🚀 Features

* 📥 PDF ingestion with OCR fallback (PaddleOCR)
* ✂️ Intelligent text chunking and cleaning
* 🔎 Semantic search using FAISS
* 🧠 LLM abstraction (OpenAI + local HuggingFace)
* 🤖 Agent-based query routing (QA / summarize / extract)
* 🧪 End-to-end CLI testing pipeline

---

## 🏗️ Architecture

```
ingestion → splitting → embeddings → vectorstore → retriever → agent → tools → LLM
```

---

## 📂 Project Structure

```
agents/        → Agent orchestration and tools
rag/           → Embeddings, retriever, vector store, splitter
llm/           → LLM abstraction layer
ingestion/     → PDF loading and OCR
utils/         → Logging and utilities
test/          → End-to-end pipeline testing
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python -m test.test_rag --pdf data/sample.pdf --backend local
```

Custom query:

```bash
python -m test.test_rag --pdf data/sample.pdf --query "What is this paper about?"
```

---

## 🧠 Supported Backends

* OpenAI (GPT models)
* Local HuggingFace (Flan-T5)

---

## 📊 Current Limitations

* Basic semantic retrieval (no reranking)
* No hybrid search (BM25 + embeddings)
* Naive context construction
* Limited prompt structuring
* No evaluation metrics (yet)

---

## 🔮 Future Improvements

* Reranking (cross-encoder / LLM-based)
* Metadata-aware retrieval
* Hybrid search
* Structured prompting with citations
* Evaluation framework (accuracy, hallucination detection)
* Section-aware chunking

---

## 📜 License

MIT License
