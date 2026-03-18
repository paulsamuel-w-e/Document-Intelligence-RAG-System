# 📄 Document Intelligence RAG System

A modular **Retrieval-Augmented Generation (RAG)** system for document understanding tasks such as:

* Question Answering (QA)
* Summarization
* Key Information Extraction

Built with a clean, extensible architecture using FAISS, sentence-transformers, and pluggable LLM backends.

---

## 🚀 Features

* 📥 PDF ingestion with OCR fallback (PaddleOCR)
* ✂️ Intelligent text cleaning and chunking
* 🔎 Semantic search using FAISS
* 🧠 LLM abstraction (OpenAI + Local HuggingFace)
* 🤖 Agent-based query routing (QA / summarize / extract)
* 🧪 End-to-end CLI pipeline

---

## 🏗️ Architecture

```
PDF → Ingestion → Chunking → Embeddings → Vector Store → Retriever → Agent → Tools → LLM → Output
```

---

## 📂 Project Structure

```
agents/        → Agent orchestration and tools
rag/           → Embeddings, retriever, vector store, splitter
llm/           → LLM abstraction layer
ingestion/     → PDF loading and OCR
utils/         → Logging utilities
test/          → End-to-end pipeline
docs/          → Detailed documentation
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python -m test.test_rag --pdf data/sample.pdf
```

Custom query:

```bash
python -m test.test_rag --pdf data/sample.pdf --query "What is this paper about?"
```

---

## 🧠 Supported LLMs

* OpenAI (GPT models)
* Local HuggingFace (Flan-T5)

---

## ⚠️ Current Limitations

* Basic semantic retrieval (no reranking)
* No hybrid search (BM25 + embeddings)
* Naive context construction
* No evaluation framework

---

## 🔮 Roadmap

* Reranking (cross-encoder / LLM)
* Metadata-aware retrieval
* Hybrid search
* Structured prompting with citations
* Evaluation metrics and benchmarking

---

## 📜 License

MIT License
