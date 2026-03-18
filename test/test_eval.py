import argparse
import json

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
    agent = build_pipeline(pdf_path, backend)
    retriever = agent._retriever  # access for evaluation

    evaluator = Evaluator()
    data = load_eval_data("eval/eval_data.json")

    results = []
    difficulty_scores = {}

    for item in data:
        query = item["query"]
        expected = item["expected_keywords"]
        difficulty = item.get("difficulty", "unknown")

        # --- Retrieval ---
        chunks, meta = retriever.retrieve_with_metadata(query)
        context_text = " ".join(m["text"] for m in meta)

        # --- Answer ---
        answer = agent.run(query)

        # --- Evaluation ---
        result = evaluator.evaluate(
            query=query,
            answer=answer,
            expected_keywords=expected,
            context=context_text,
            retrieval_meta=meta,
        )

        results.append(result)

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

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    avg_answer = sum(r["answer_score"] for r in results) / len(results)
    avg_retrieval = sum(r["retrieval_score"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Average Answer Score: {avg_answer:.2f}")
    print(f"Average Retrieval Score: {avg_retrieval:.2f}")

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
        choices=["local", "openai"],
        help="LLM backend",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_evaluation(args.pdf, args.backend)


if __name__ == "__main__":
    main()