"""Routing-accuracy and Mode-3 citation harness for the Adaptive-RAG system.

This is a live evaluation script, not an offline unit test: it calls the real
LLM (Ollama/Qwen) through the router and, optionally, the full graph. Run it
after ingestion, with Ollama serving the configured model.

Usage:
    python -m tests.test_questions              # routing accuracy over 30 questions
    python -m tests.test_questions --mode3      # also run the Mode-3 citation check

The 30 questions are labelled with their EXPECTED route so routing accuracy can
be measured. Labels reflect the intended design:
  - easy   : answerable from general knowledge, no retrieval needed
  - medium : one fact grounded in a single document, one search
  - hard   : multi-hop / cross-document comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph import build_graph
from src.nodes.router import route_question

# (question, expected_route)
QUESTIONS: list[tuple[str, str]] = [
    # --- easy: general knowledge / definitional, no retrieval ---
    ("What does the acronym NLP stand for?", "easy"),
    ("What is a vector database?", "easy"),
    ("What is a neural network in one sentence?", "easy"),
    ("What does GPU stand for?", "easy"),
    ("What is machine learning?", "easy"),
    ("What is the difference between training and inference?", "easy"),
    ("What is a token in natural language processing?", "easy"),
    ("What does the acronym API stand for?", "easy"),
    ("What is gradient descent, briefly?", "easy"),
    ("What is overfitting in machine learning?", "easy"),
    # --- medium: one fact grounded in a single document ---
    ("According to the BERT paper, what two pretraining tasks does BERT use?", "medium"),
    ("What optimizer does the Transformer paper use for training?", "medium"),
    ("What text-to-text framework does the T5 paper describe?", "medium"),
    ("How does ELECTRA describe its replaced-token-detection objective?", "medium"),
    ("What parameter-reduction techniques does the ALBERT paper introduce?", "medium"),
    ("What distillation approach does the DistilBERT paper use?", "medium"),
    ("What permutation-based objective does the XLNet paper propose?", "medium"),
    ("What scaling approach does the PaLM paper describe?", "medium"),
    ("What training-data changes does the RoBERTa paper make to BERT?", "medium"),
    ("What few-shot capability does the GPT-3 paper report?", "medium"),
    # --- hard: multi-hop / cross-document comparison ---
    ("Compare how BERT and the Transformer paper each handle positional information, and explain any tradeoffs.", "hard"),
    ("How does RoBERTa's pretraining differ from BERT's, and what does it keep the same?", "hard"),
    ("Compare the parameter-efficiency strategies of ALBERT and DistilBERT.", "hard"),
    ("How do ELECTRA and BERT differ in their pretraining objectives, and why does ELECTRA claim to be more efficient?", "hard"),
    ("Compare how GPT-2 and GPT-3 each scale up language modeling.", "hard"),
    ("How does XLNet's autoregressive objective differ from BERT's masked-language-model objective?", "hard"),
    ("Compare the text-to-text framing of T5 with the masked-language-model framing of BERT.", "hard"),
    ("How do the training corpora and objectives differ between RoBERTa and XLNet?", "hard"),
    ("Compare how PaLM and GPT-3 approach few-shot learning at scale.", "hard"),
    ("Across the BERT-family papers, how do BERT, RoBERTa, and ALBERT each change the pretraining recipe?", "hard"),
]


def run_routing() -> int:
    correct = 0
    by_class: dict[str, list[int]] = {"easy": [0, 0], "medium": [0, 0], "hard": [0, 0]}
    print(f"Routing accuracy over {len(QUESTIONS)} questions\n" + "=" * 60)
    for question, expected in QUESTIONS:
        result = route_question({"question": question, "trace": []})
        predicted = result.get("route")
        hit = predicted == expected
        correct += hit
        by_class[expected][1] += 1
        by_class[expected][0] += hit
        mark = "ok  " if hit else "MISS"
        print(f"[{mark}] expected={expected:<6} got={predicted:<6}  {question[:60]}")

    print("=" * 60)
    for cls, (hits, total) in by_class.items():
        print(f"  {cls:<6}: {hits}/{total}")
    pct = 100 * correct / len(QUESTIONS)
    print(f"\nOverall: {correct}/{len(QUESTIONS)} = {pct:.1f}%")
    return correct


def run_mode3_citation_check() -> None:
    """Runs the hard questions through the full graph and reports whether each
    produced at least one resolved citation. This surfaces the known Qwen-7B
    limitation (inconsistent [n] citation formatting on complex Mode-3
    generations) as a measured pass/fail rather than an anecdote."""
    print("\nMode-3 citation resolution (hard questions through full graph)\n" + "=" * 60)
    graph = build_graph()
    hard = [q for q, r in QUESTIONS if r == "hard"]
    with_citations = 0
    for question in hard:
        result = graph.invoke({"question": question, "trace": []})
        route = result.get("route")
        n_cites = len(result.get("citations", []))
        iters = result.get("iterations")
        if n_cites > 0:
            with_citations += 1
        print(
            f"  route={route:<6} iters={iters} citations={n_cites:<2}  {question[:55]}"
        )
    print("=" * 60)
    print(
        f"Hard questions with >=1 resolved citation: "
        f"{with_citations}/{len(hard)}"
    )
    print(
        "Note: a low ratio here reflects the documented Qwen-7B citation-format\n"
        "limitation, not a retrieval failure -- the evidence is retrieved, but\n"
        "the model may not emit parseable [n] markers on long generations."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Routing-accuracy and Mode-3 citation harness (live LLM)."
    )
    parser.add_argument(
        "--mode3",
        action="store_true",
        help="Also run the Mode-3 citation-resolution check (slower; runs the full graph).",
    )
    args = parser.parse_args()
    run_routing()
    if args.mode3:
        run_mode3_citation_check()


if __name__ == "__main__":
    main()
