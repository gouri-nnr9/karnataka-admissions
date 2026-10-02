from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

project_dir = Path(__file__).parent
processed_dir = project_dir / "data" / "processed"

chunks = json.loads(
    (processed_dir / "chunks_embedded.json").read_text(
        encoding="utf-8"
    )
)

test_cases = json.loads(
    (project_dir / "evaluation" / "questions.json").read_text(
        encoding="utf-8"
    )
)

embedding_model = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    device="cpu",
)

document_vectors = np.array(
    [chunk["embedding"] for chunk in chunks],
    dtype=np.float32,
)

TOP_K = 100

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
    query_vector = embedding_model.encode(
    "query: " + question,
    normalize_embeddings=True,
    )
    similarities = document_vectors @ query_vector
    all_ranked_indices = np.argsort(similarities)[::-1]
    if case["answerable"]:
        expected_ids = set(case["expected_chunk_ids"])
        expected_ranks = []
        for rank, index in enumerate(all_ranked_indices, start=1):
            chunk_id = chunks[int(index)]["chunk_id"]
            if chunk_id in expected_ids:
                expected_ranks.append(rank)
        print(f"Expected chunk vector rank: {expected_ranks}")
    candidate_indices = np.argsort(similarities)[::-1][:TOP_K]

    candidates = [
    chunks[int(index)]
    for index in candidate_indices
    ]

    pairs = [
    (question, chunk["text"])
    for chunk in candidates
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
    candidates[int(index)]["chunk_id"]
    for index in top_indices
    ]

    for rank, index in enumerate(top_indices, start=1):
        print(
            f"{rank}. {candidates[int(index)]['chunk_id']} "
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


print("\nEmbedding baseline: Hit@1 = 20%, Hit@3 = 20%")

if answerable_count:
    print(f"Reranker Hit@1: {hits_at_1 / answerable_count:.0%}")
    print(f"Reranker Hit@3: {hits_at_3 / answerable_count:.0%}")

output_path = project_dir / "evaluation" / "reranker_results.json"
output_path.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"Saved results: {output_path}")