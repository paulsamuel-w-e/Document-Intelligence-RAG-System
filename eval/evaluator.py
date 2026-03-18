# eval/evaluator.py
from typing import List, Dict, Optional
from difflib import SequenceMatcher
import re


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


class Evaluator:
    """
    Evaluator that:
      - Scores answers by keyword hit-rate (exact + fuzzy)
      - Scores retrieval by presence of keywords in retrieved chunks
      - Performs a simple hallucination check (answer text contained in context)
      - Returns detailed diagnostic info for debugging
    """

    def __init__(self, fuzzy_threshold: float = 0.75) -> None:
        self.fuzzy_threshold = fuzzy_threshold

    def _fuzzy_contains(self, text: str, keyword: str) -> bool:
        # quick fuzzy check using SequenceMatcher ratio for longer phrases,
        # and substring check for short tokens
        t = _normalize_text(text)
        k = _normalize_text(keyword)
        if k in t:
            return True
        # if keyword longer than 6 chars, do ratio check against sliding windows
        words = t.split()
        if len(k) > 6:
            window = len(k.split())
            if window <= 0:
                window = 1
            # join successive windows to compare
            for i in range(0, max(1, len(words) - window + 1)):
                piece = " ".join(words[i : i + window + 4])  # slightly larger window
                ratio = SequenceMatcher(None, piece, k).ratio()
                if ratio >= self.fuzzy_threshold:
                    return True
        return False

    def keyword_match_score(self, answer: str, keywords: List[str]) -> float:
        if not keywords:
            return 0.0
        hits = 0
        for kw in keywords:
            if self._fuzzy_contains(answer, kw):
                hits += 1
        return hits / len(keywords)

    def retrieval_score(self, retrieval_meta: List[Dict], keywords: List[str]) -> float:
        """
        Check how many expected keywords appear in the retrieved chunks.
        retrieval_meta: list of dicts with at least 'text' and optionally 'section'/'score'
        """
        if not keywords:
            return 0.0
        combined = " ".join(m.get("text", "") for m in retrieval_meta)
        hits = sum(1 for kw in keywords if self._fuzzy_contains(combined, kw))
        return hits / len(keywords)

    def hallucination_check(self, answer: str, context: Optional[str]) -> bool:
        """
        Very simple check: does the answer text (or significant keywords)
        appear in the provided context? If not, flag as possible hallucination.
        This is conservative and only a heuristic.
        """
        if not context:
            return True  # can't verify → treat as risky
        a = _normalize_text(answer)
        c = _normalize_text(context)
        # if large portion of answer present in context, it's not hallucination
        if a and a in c:
            return False
        # check by sentences / keyword overlap
        return True  # likely hallucination if not directly found

    def evaluate(
        self,
        query: str,
        answer: str,
        expected_keywords: List[str],
        context: Optional[str] = None,
        retrieval_meta: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Returns a diagnostic dict:
          - answer_score: keyword hit rate in answer
          - retrieval_score: keyword hit rate in retrieved chunks
          - hallucination: boolean (True = suspicious)
          - missing_keywords: list[str] keywords not found in answer
          - evidence: short excerpt from context where keywords appeared (if any)
        """
        answer_score = self.keyword_match_score(answer, expected_keywords)
        retrieval_score = 0.0
        if retrieval_meta is not None:
            retrieval_score = self.retrieval_score(retrieval_meta, expected_keywords)

        missing = [kw for kw in expected_keywords if not self._fuzzy_contains(answer, kw)]

        # extract small evidence snippets (first chunk where keyword found)
        evidence = {}
        if retrieval_meta:
            combined_chunks = retrieval_meta
            for kw in expected_keywords:
                for m in combined_chunks:
                    if self._fuzzy_contains(m.get("text", ""), kw):
                        evidence.setdefault(kw, []).append(
                            {"chunk_index": m.get("index"), "section": m.get("section"), "snippet": m.get("text")[:200]}
                        )
                        break

        hall = self.hallucination_check(answer, context)

        return {
            "query": query,
            "answer": answer,
            "answer_score": answer_score,
            "retrieval_score": retrieval_score,
            "passed": answer_score >= 0.5,
            "missing_keywords": missing,
            "evidence": evidence,
            "hallucination_suspected": hall,
        }