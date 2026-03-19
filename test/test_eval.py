import argparse
import json
import time

from eval.evaluator import Evaluator
from test.test_rag import build_pipeline


# ---------------------------------------------------------------------------
# Load eval data
# ---------------------------------------------------------------------------

def load_eval_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------

def run_evaluation(pdf_path: str, backend: str):
    """
    Run full evaluation pipeline:
    - Builds RAG system
    - Executes queries
    - Computes retrieval + answer metrics
    - Reports aggregated results
    """
    if backend == "llama_cpp":
        agent = build_pipeline(
            pdf_path,
            backend,
        )
    else:
        agent = build_pipeline(pdf_path, backend)
    retriever = agent._retriever  # access for evaluation

    evaluator = Evaluator()
    data = load_eval_data("eval/eval_data.json")

    results = []
    difficulty_scores = {}

    print(f"\nRunning evaluation with backend: {backend}")

    for item in data:
        query = item["query"]
        expected = item["expected_keywords"]
        difficulty = item.get("difficulty", "unknown")

        # --- Retrieval ---
        chunks = retriever.retrieve(query)
        context_text = " ".join(chunks)
        meta = [{"text": c} for c in chunks]
        context_text = "\n\n".join(
            f"[Chunk {i+1}] {m['text']}"
            for i, m in enumerate(meta)
        )

        # --- Answer ---
        start = time.perf_counter()
        answer = agent.run(query)
        latency = time.perf_counter() - start

        # --- Evaluation ---
        result = evaluator.evaluate(
            query=query,
            answer=answer,
            expected_keywords=expected,
            context=context_text,
            retrieval_meta=meta,
        )

        results.append(result)
        latencies = []
        latencies.append(latency)

        # --- Difficulty tracking ---
        difficulty_scores.setdefault(difficulty, []).append(result["answer_score"])

        # --- Print ---
        print("\n" + "=" * 60)
        print(f"Query: {query}")
        print(f"Difficulty: {difficulty}")
        print(f"Answer: {answer}")
        print(f"Retrieval Score: {result['retrieval_score']:.2f}")
        print(f"Answer Score: {result['answer_score']:.2f}")
        print(f"Hallucination Suspected: {result['hallucination_suspected']}")
        print(f"Missing Keywords: {result['missing_keywords']}")
        print(f"Passed: {result['passed']}")
        print(f"Latency: {latency:.3f}s")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    avg_answer = sum(r["answer_score"] for r in results) / len(results)
    avg_retrieval = sum(r["retrieval_score"] for r in results) / len(results)
    avg_latency = sum(latencies) / len(latencies)

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Average Answer Score: {avg_answer:.2f}")
    print(f"Average Retrieval Score: {avg_retrieval:.2f}")
    print(f"Average Latency: {avg_latency:.3f}s")

    print("\n--- By Difficulty ---")
    for diff, scores in difficulty_scores.items():
        avg = sum(scores) / len(scores)
        print(f"{diff}: {avg:.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation on RAG system.")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument(
        "--backend",
        default="local",
        choices=["local", "openai", "llama_cpp"],
        help="LLM backend",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_evaluation(args.pdf, args.backend)


if __name__ == "__main__":
    main()