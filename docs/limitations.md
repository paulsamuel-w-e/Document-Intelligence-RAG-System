⚠️ Current Limitations

1. LLM Layer

* Local model (Flan-T5) has weak reasoning ability
* Poor handling of negation and multi-step reasoning
* Occasional copying of context patterns

---

2. Prompting / Generation Control

* Prompts are not strictly enforced
* Model may copy context artifacts (e.g., [Chunk X])
* Weak control over output structure
* Answers may be incomplete or overly short

---

3. Context Construction (Major Limitation)

* Flat concatenation of chunks
* No hierarchical structure
* No prioritization (abstract > intro > body)
* No global document awareness

---

4. Answer Quality

* Often extractive instead of explanatory
* Missing key concepts despite correct retrieval
* Weak synthesis across multiple chunks

---

5. Chunking

* Character-based splitting
* Breaks semantic boundaries
* Not heading-aware

---

6. Agent Layer

* Heuristic routing (keyword-based)
* Limited handling of complex queries

---

7. Evaluation

* Keyword-based scoring (approximate)
* Simple hallucination heuristic
* No semantic grading

---

8. System Engineering

* No caching (OCR / embeddings / LLM)
* No streaming responses
* No latency optimization

---

🧠 Key Insight

The system’s primary bottleneck is:

→ LLM reasoning and synthesis, not retrieval

Retrieval performance is strong (~0.67),
but answer quality remains low (~0.22).
