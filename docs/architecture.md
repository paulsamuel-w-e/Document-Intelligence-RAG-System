# 🏗️ System Architecture

## Overview

The system follows a modular RAG pipeline:

```
Ingestion → Processing → Embedding → Storage → Retrieval → Generation
```

---

## Layers

### 1. Ingestion Layer

* Extracts text from PDFs using PyMuPDF
* Falls back to OCR using PaddleOCR

---

### 2. Processing Layer

* Cleans text (removes noise)
* Splits into chunks using recursive splitting

---

### 3. Embedding Layer

* Converts text chunks into dense vectors
* Uses sentence-transformers

---

### 4. Storage Layer

* FAISS vector store (IndexFlatIP)
* Stores embeddings + raw text

---

### 5. Retrieval Layer

* Converts query → embedding
* Retrieves top-k relevant chunks

---

### 6. Agent Layer

* Detects user intent
* Routes to appropriate tool

---

### 7. Tools Layer

* Task-specific prompt construction:

  * QA
  * Summarization
  * Extraction

---

### 8. LLM Layer

* Generates final response
* Supports OpenAI + local models

---

## Design Principles

* Modular and extensible
* Clear separation of concerns
* Replaceable components
