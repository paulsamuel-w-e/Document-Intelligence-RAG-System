# 🚀 Roadmap

## ⚡ Phase 1 — Controlled Reasoning (Immediate Priority)

* Implement two-step generation:

  * Step 1: Extract evidence
  * Step 2: Generate answer
* Add negation-aware prompting:

  * Explicit handling of “NOT” queries
* Enforce strict grounding:

  * No external knowledge
  * Evidence-based answers only
* Improve binary question handling (yes/no with justification)

---

## ⚡ Phase 2 — Context Intelligence

* Structured context:

  * section-aware grouping
  * prioritized ordering
* Context compression (remove redundancy)
* Better chunk selection strategies

---

## ⚡ Phase 3 — Answer Quality Optimization

* Keyword anchoring (ensure coverage of key concepts)
* Structured outputs:

  * facts → reasoning → answer
* Improve explanation depth

---

## ⚡ Phase 4 — Model Upgrade

* Switch to stronger LLM (OpenAI)
* Compare reasoning vs local model
* Evaluate cost-performance trade-offs

---

## ⚡ Phase 5 — Retrieval Refinement

* Semantic chunking (heading-aware)
* Adaptive retrieval (dynamic k, thresholds)
* Improved reranking strategies

---

## ⚡ Phase 6 — Evaluation Enhancements

* LLM-based grading (semantic correctness)
* Faithfulness scoring
* Retrieval recall@k

---

## ⚡ Phase 7 — System Engineering

* Caching (OCR, embeddings, LLM)
* Streaming responses
* Latency optimization

---

## ⚡ Phase 8 — Advanced Agent

* LLM-based routing
* Multi-step reasoning pipelines
* Tool chaining

---

# 🎯 Strategic Direction

The system evolution is guided by:

> Retrieval → solved
> Reasoning → current bottleneck
> Control → next frontier
