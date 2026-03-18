# ⚠️ Current Limitations

## 1. LLM Layer

* Weak local model (Flan-T5)
* No structured prompting
* No citation enforcement

---

## 2. Retrieval

* No reranking
* Fixed top-k
* No score filtering
* No hybrid search

---

## 3. Context Handling

* Naive concatenation
* No metadata
* No prioritization of important sections
* No global context awareness

---

## 4. Chunking

* Character-based splitting
* Breaks semantic structure
* Not section-aware

---

## 5. Agent Layer

* Keyword-based intent detection
* Not robust for complex queries

---

## 6. Ingestion

* No caching
* No partial OCR
* Full OCR fallback only

---

## 7. System

* No streaming
* No latency handling
* No caching

---

## 8. Evaluation

* No answer correctness metrics
* No retrieval evaluation
* No hallucination detection
