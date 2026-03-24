# ⚠️ Current Limitations

## 1. Reasoning Control (Primary Bottleneck)

* Weak handling of negation ("NOT", absence-based queries)
* No explicit reasoning steps
* Model may guess instead of verifying

---

## 2. Answer Precision

* Tendency to produce hedged or non-committal answers
* Inconsistent handling of binary (yes/no) questions

---

## 3. Context Utilization

* Occasionally uses general knowledge instead of context
* Weak enforcement of “context-only” constraint

---

## 4. Keyword Alignment

* Answers may be correct but miss expected key terms
* Impacts evaluation scores despite correctness

---

## 5. Technical Depth

* Limited ability to provide concept-complete explanations
* Struggles with detailed technical breakdowns

---

## 6. Context Construction

* Flat chunk concatenation
* No hierarchy or prioritization
* No explicit relationship between chunks

---

## 7. Evaluation Limitations

* Keyword-based scoring (approximate)
* Does not capture semantic correctness fully

---

## 🧠 Key Insight

The system has transitioned from a retrieval bottleneck to a reasoning bottleneck:

→ Retrieval is strong (~0.77)
→ Answer quality is improved (~0.71)
→ Remaining failures are due to reasoning and control limitations
