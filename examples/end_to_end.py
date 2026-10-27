"""
End-to-end walkthrough of the acclaim pipeline on a single answer.

Runs evaluate() with the bundled default config and prints the output of
every pipeline stage. See examples/README.md for an annotated version.

Usage (needs OPENROUTER_API_KEY for the default config):

    python examples/end_to_end.py
"""

from acclaim import Document, evaluate, load_config

question = "When was the Eiffel Tower built and how tall is it?"

documents = [
    Document(
        doc_id="eiffel_history",
        text=(
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
            "in Paris, France. It was designed by Gustave Eiffel's engineering "
            "company and constructed from 1887 to 1889 as the centrepiece of the "
            "1889 World's Fair."
        ),
    ),
    Document(
        doc_id="eiffel_facts",
        text=(
            "The tower is 330 metres tall, about the same height as an 81-storey "
            "building. It was the tallest man-made structure in the world until the "
            "Chrysler Building in New York City was finished in 1930."
        ),
    ),
]

answer = (
    "The Eiffel Tower was built between 1887 and 1889 for the World's Fair [1]. "
    "It is 300 metres tall [2]. "
    "It remained the tallest structure in the world until 1930 [3]. "
    "It is located on the Champ de Mars in Paris. "
    "It is arguably the most beautiful monument in Europe [1]."
)


def main() -> None:
    config = load_config()  # bundled acclaim/config/default.yaml
    result = evaluate(answer, documents, question=question, config=config)
    steps = result.steps

    print("── 1. Claim extraction ─────────────────────────────")
    for c in steps["claims"]:
        print(f"  {c.text!r}  span={c.span}")

    print("\n── 2. Claim filtering ──────────────────────────────")
    for d in steps["filter_decisions"]:
        verdict = "keep" if d.keep else "DROP"
        print(f"  [{verdict}] {d.claim.text!r}  ({d.category}: {d.reason})")

    print("\n── 3. Citation alignment ───────────────────────────")
    print(f"  citation_to_doc = {steps['citation_to_doc']}")
    for a in steps["aligned_claims"]:
        print(f"  {a.claim.text!r} -> {a.citation_doc_ids}")

    print("\n── 4. Evidence judgement ───────────────────────────")
    for cr in steps["claim_results"]:
        s = cr.support
        print(f"  {cr.claim.text!r}")
        print(f"    {s.label.value} (confidence={s.confidence:.2f}): {s.reason}")

    print("\n── 5. Metrics ──────────────────────────────────────")
    for name, value in result.metrics.items():
        print(f"  {name}: {value:.4f}")

    result.to_jsonl("end_to_end_result.jsonl")
    print("\nWrote end_to_end_result.jsonl")


if __name__ == "__main__":
    main()
