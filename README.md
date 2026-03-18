📄 Document Intelligence RAG System

A modular, production-oriented Retrieval-Augmented Generation (RAG) system for document understanding tasks such as:

* Question Answering (QA)
* Summarization
* Key Information Extraction

The system implements **hybrid retrieval (dense + sparse), metadata-aware filtering, and evaluation-driven development**.

---

🚀 Features

📥 PDF ingestion with OCR fallback (PaddleOCR)
✂️ Text cleaning and chunking with noise filtering
🔎 Hybrid retrieval (FAISS + BM25)
🎯 Metadata-aware ranking (section-based weighting)
🧠 LLM abstraction (OpenAI + Local HuggingFace)
🤖 Agent-based query routing with query-type awareness
📊 Evaluation framework (retrieval + answer quality + hallucination signals)
🧪 CLI-based testing and benchmarking

---

🏗️ Architecture

PDF
→ Ingestion
→ Cleaning + Chunking
→ Embeddings (dense)
→ Vector Store (FAISS)
→ Hybrid Retrieval (Dense + BM25)
→ Metadata Filtering
→ Reranking
→ Agent (routing + strategy)
→ Tools (QA / Summarize / Extract)
→ LLM
→ Output

---

📂 Project Structure

agents/        → Agent orchestration and tools
rag/           → Embeddings, retriever (hybrid), vector store, splitter
llm/           → LLM abstraction layer
ingestion/     → PDF loading and OCR
eval/          → Evaluation dataset + evaluator
utils/         → Logging utilities
test/          → CLI pipeline and evaluation scripts
docs/          → System documentation

---

⚙️ Installation

pip install -r requirements.txt

---

▶️ Usage

Run single query:

python -m test.test_rag --pdf data/sample.pdf --query "What is this paper about?"

Run evaluation:

python -m test.test_eval --pdf data/sample.pdf --backend local

---

🧠 Supported LLMs

* OpenAI (GPT models)
* Local HuggingFace (Flan-T5)

---

📊 Evaluation

The system includes a dataset-driven evaluation pipeline measuring:

* Retrieval quality (keyword recall in retrieved chunks)
* Answer quality (keyword coverage in generated response)
* Hallucination signals (context grounding checks)
* Difficulty-wise performance (easy / medium / hard / tricky)

---

⚠️ Current Limitations

* Local LLM (Flan-T5) limits reasoning quality
* Context construction is still flat (no hierarchical structure)
* Prompt control is still basic (structured generation not fully enforced)
* No citation validation yet
* No semantic chunking (still character-based)
* Agent routing is heuristic (not LLM-driven)

---

🔮 Roadmap

* Structured context construction (section-aware)
* Stronger prompt control and grounded generation
* Semantic chunking (heading-aware)
* LLM-based routing
* Production optimizations (caching, streaming)

---

📜 License

MIT License
