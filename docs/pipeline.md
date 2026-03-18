# 🔄 End-to-End Pipeline

## Step-by-Step Flow

### 1. Document Loading

* Input: PDF file
* Output: raw text

---

### 2. Text Extraction

* PyMuPDF extracts text
* OCR fallback if quality is low

---

### 3. Text Cleaning

* Removes:

  * references
  * captions
  * page numbers

---

### 4. Chunking

* Chunk size: 500
* Overlap: 100
* Filters noisy chunks

---

### 5. Embedding Generation

* Converts chunks into vectors
* Normalized for cosine similarity

---

### 6. Vector Storage

* Stored in FAISS index
* Supports similarity search

---

### 7. Query Processing

#### a. Intent Detection

* Keyword-based routing

#### b. Retrieval

* Top-k chunks retrieved

#### c. Tool Execution

* Prompt constructed

#### d. LLM Generation

* Final response produced

---

## Output

* Answer / Summary / Extracted facts
