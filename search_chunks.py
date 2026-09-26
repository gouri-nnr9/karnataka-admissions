from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer

project_dir = Path(__file__).parent
processed_dir = project_dir / "data" / "processed"

# Load the nine English bell-timing chunks.
chunks = json.loads(
    (processed_dir / "bell_timings_embedded.json").read_text(
        encoding="utf-8"
    )
)

# Add the Kannada notification we embedded earlier.
notice = json.loads(
    (processed_dir / "notification_25092026_embedded.json").read_text(
        encoding="utf-8"
    )
)
chunks.append(notice)

model_name = "intfloat/multilingual-e5-small"

for chunk in chunks:
    if chunk["embedding_model"] != model_name:
        raise ValueError(
            f"Embedding model mismatch: {chunk['chunk_id']}"
        )

model = SentenceTransformer(model_name, device="cpu")

# Arrange the saved embeddings into one matrix.
document_vectors = np.array(
    [chunk["embedding"] for chunk in chunks],
    dtype=np.float32,
)

# questions = [
#     "When were the PGCET 2026 first-round results published?",
#     "When will the choice entry and fee payment schedule be published?",
#     "When can M.Tech candidates enter the examination hall?",
#     "When should M.Tech candidates start answering the questions?",
#     "Are calculators allowed in the M.Tech examination?",
#     "How do I bake a chocolate cake?",
# ]
test_cases = json.loads(
    (project_dir / "evaluation" / "questions.json").read_text(
        encoding="utf-8"
    )
)

answerable_count = 0
hits_at_1 = 0
hits_at_3 = 0

print(f"\nLoaded {len(chunks)} chunks.")

# for question in questions:
#     query_vector = model.encode(
#         "query: " + question,
#         normalize_embeddings=True,
#     )

#     # Compare the question against every saved chunk.
#     # For normalized vectors, dot product = cosine similarity.
#     scores = document_vectors @ query_vector

#     # Sort scores and select the three highest.
#     top_indices = np.argsort(scores)[::-1][:3]

#     print(f"\nQuestion: {question}")

#     for rank, index in enumerate(top_indices, start=1):
#         chunk = chunks[int(index)]
#         metadata = chunk["metadata"]

#         print(
#             f"{rank}. {chunk['chunk_id']} "
#             f"| similarity: {scores[index]:.4f}"
#         )
#         print(
#             f"   Source: {metadata['source_file']} "
#             f"| page: {metadata['page']}"
#         )
for case in test_cases:
    question = case["question"]

    query_vector = model.encode(
        "query: " + question,
        normalize_embeddings=True,
    )

    scores = document_vectors @ query_vector
    top_indices = np.argsort(scores)[::-1][:3]

    retrieved_ids = [
        chunks[int(index)]["chunk_id"]
        for index in top_indices
    ]

    print(f"\nQuestion: {question}")

    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[int(index)]
        print(
            f"{rank}. {chunk['chunk_id']} "
            f"| similarity: {scores[index]:.4f}"
        )

    if case["answerable"]:
        answerable_count += 1
        expected_ids = set(case["expected_chunk_ids"])

        found_at_1 = retrieved_ids[0] in expected_ids
        found_at_3 = bool(expected_ids.intersection(retrieved_ids))

        hits_at_1 += int(found_at_1)
        hits_at_3 += int(found_at_3)

        print(f"Expected chunk ranked first: {found_at_1}")
        print(f"Expected chunk in top three: {found_at_3}")
    else:
        print("Unsupported question — refusal handling not implemented.")

if answerable_count:
    print("\nRetrieval evaluation:")
    print(
        f"Hit@1: {hits_at_1}/{answerable_count} "
        f"({hits_at_1 / answerable_count:.0%})"
    )
    print(
        f"Hit@3: {hits_at_3}/{answerable_count} "
        f"({hits_at_3 / answerable_count:.0%})"
    )