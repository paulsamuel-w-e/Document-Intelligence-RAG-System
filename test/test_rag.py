"""
End-to-end integration test for the Document Intelligence System.

Usage:
    python test_rag.py --pdf path/to/document.pdf --backend openai
    python test_rag.py --pdf path/to/document.pdf --backend local

The script:
  1. Loads and extracts text from a PDF.
  2. Cleans and chunks the text.
  3. Builds a FAISS vector store.
  4. Runs a series of test queries through the agent.
  5. Prints results to stdout.
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure project root is on the Python path when run directly
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from agents.agent import DocumentAgent
from ingestion.loader import load_document
from llm.llm_wrapper import get_llm
from rag.embeddings import EmbeddingModel
from rag.retriever import Retriever
from rag.splitter import split_text
from rag.vectorstore import VectorStore
from utils.logger import get_logger

logger = get_logger("test_rag")

TEST_QUERIES = [
    "What is the main topic of this document?",
    "Summarize the document.",
    "Extract the key facts and entities.",
]


def build_pipeline(pdf_path: str, backend: str, top_k: int = 5):
    """
    Build and return a DocumentAgent from a PDF file.

    Steps:
      - Load document text
      - Split into chunks
      - Embed chunks
      - Store in FAISS
      - Wrap in Retriever + Agent
    """
    # 1. Ingestion
    logger.info("=== Step 1: Document Ingestion ===")
    text = load_document(pdf_path)
    logger.info("Document loaded. Total characters: %d", len(text))

    # 2. Chunking
    logger.info("=== Step 2: Text Splitting ===")
    chunks = split_text(text)
    logger.info("Total chunks: %d", len(chunks))

    if not chunks:
        raise RuntimeError("No valid chunks extracted. Check the PDF quality.")

    # 3. Embeddings
    logger.info("=== Step 3: Embedding ===")
    embed_model = EmbeddingModel()
    embeddings = embed_model.encode(chunks)
    logger.info("Embeddings shape: %s", embeddings.shape)

    # 4. Vector store
    logger.info("=== Step 4: Building Vector Store ===")
    store = VectorStore(dimension=embed_model.dimension)
    store.add(chunks, embeddings)
    logger.info("Vector store size: %d", store.size)

    # 5. Retriever
    retriever = Retriever(embed_model, store, top_k=top_k)

    # 6. LLM
    logger.info("=== Step 5: Loading LLM (backend=%s) ===", backend)
    llm = get_llm(backend=backend)

    # 7. Agent
    agent = DocumentAgent(retriever=retriever, llm=llm)

    return agent


def run_queries(agent: DocumentAgent, queries: list[str]) -> None:
    """Run each query through the agent and print the result."""
    print("\n" + "=" * 70)
    print("DOCUMENT INTELLIGENCE SYSTEM — TEST RESULTS")
    print("=" * 70)

    for i, query in enumerate(queries, start=1):
        print(f"\n[Query {i}] {query}")
        print("-" * 50)
        try:
            answer = agent.run(query)
            print(f"Answer:\n{answer}")
        except Exception as exc:  # noqa: BLE001
            logger.error("Query failed: %s", exc)
            print(f"ERROR: {exc}")

    print("\n" + "=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end test for the Document Intelligence System."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the PDF document to process.",
    )
    parser.add_argument(
        "--backend",
        default="openai",
        choices=["openai", "local"],
        help="LLM backend to use (default: openai).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query (default: 5).",
    )
    parser.add_argument(
        "--query",
        nargs="*",
        help="Custom queries to run (overrides default test queries).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    queries = args.query if args.query else TEST_QUERIES

    agent = build_pipeline(
        pdf_path=args.pdf,
        backend=args.backend,
        top_k=args.top_k,
    )

    run_queries(agent, queries)


if __name__ == "__main__":
    main()