from pathlib import Path
import json

import numpy as np
from sentence_transformers import CrossEncoder

project_dir = Path(__file__).parent
processed_dir = project_dir / "data" / "processed"

chunks = json.loads(
    (processed_dir / "bell_timings_embedded.json").read_text(
        encoding="utf-8"
    )
)

notice = json.loads(
    (processed_dir / "notification_25092026_embedded.json").read_text(
        encoding="utf-8"
    )
)
chunks.append(notice)

test_cases = json.loads(
    (project_dir / "evaluation" / "questions.json").read_text(
        encoding="utf-8"
    )
)

print("Loading reranker...")
reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    device="cpu",
    max_length=1024,
)

hits_at_1 = 0
hits_at_3 = 0
answerable_count = 0
results = []

for case in test_cases:
    question = case["question"]
    print(f"\nQuestion: {question}", flush=True)

    # For this small experiment, score all ten chunks.
    # Unlike E5 embeddings, these inputs need no query/passage prefixes.
    pairs = [
        (question, chunk["text"])
        for chunk in chunks
    ]

    # Check using the reranker's own tokenizer.
    for pair in pairs:
        token_count = len(
            reranker.tokenizer(
                pair[0],
                pair[1],
                truncation=False,
            )["input_ids"]
        )
        if token_count > 1024:
            raise ValueError(
                f"Question/chunk pair exceeds limit: {token_count} tokens"
            )

    scores = np.asarray(
        reranker.predict(pairs, batch_size=1)
    ).reshape(-1)

    top_indices = np.argsort(scores)[::-1][:3]
    retrieved_ids = [
        chunks[int(index)]["chunk_id"]
        for index in top_indices
    ]

    for rank, index in enumerate(top_indices, start=1):
        print(
            f"{rank}. {chunks[int(index)]['chunk_id']} "
            f"| reranker score: {scores[index]:.4f}"
        )

    if case["answerable"]:
        answerable_count += 1
        expected = set(case["expected_chunk_ids"])

        hits_at_1 += int(retrieved_ids[0] in expected)
        hits_at_3 += int(bool(expected.intersection(retrieved_ids)))
    else:
        print("Unsupported question — no refusal rule applied yet.")

    results.append({
        "id": case["id"],
        "question": question,
        "retrieved_ids": retrieved_ids,
        "scores": [float(scores[index]) for index in top_indices],
    })

print("\nEmbedding baseline: Hit@1 = 60%, Hit@3 = 80%")

if answerable_count:
    print(f"Reranker Hit@1: {hits_at_1 / answerable_count:.0%}")
    print(f"Reranker Hit@3: {hits_at_3 / answerable_count:.0%}")

output_path = project_dir / "evaluation" / "reranker_results.json"
output_path.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"Saved results: {output_path}")