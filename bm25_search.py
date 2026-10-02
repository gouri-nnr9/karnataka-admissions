from pathlib import Path
import json
import re
import numpy as np

from rank_bm25 import BM25Okapi

project_dir = Path(__file__).parent
processed_dir = project_dir / "data" / "processed"

chunks = json.loads(
    (processed_dir / "chunks.json").read_text(
        encoding="utf-8"
    )
)

print(f"Loaded {len(chunks)} chunks.")

def tokenize(text):
    return re.findall(r"\w+", text.lower())


tokenized_chunks = [
    tokenize(chunk["text"])
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)

test_cases = json.loads(
    (project_dir / "evaluation" / "questions.json").read_text(
        encoding="utf-8"
    )
)

for case in test_cases:
    question = case["question"]

    query_tokens = tokenize(question)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = np.argsort(scores)[::-1]

    print(f"\nQuestion: {question}")

    if case["answerable"]:
        expected_ids = set(case["expected_chunk_ids"])

        expected_ranks = []

        for rank, index in enumerate(ranked_indices, start=1):
            chunk_id = chunks[int(index)]["chunk_id"]

            if chunk_id in expected_ids:
                expected_ranks.append(rank)

        print(f"Expected chunk BM25 rank: {expected_ranks}")

    top_indices = ranked_indices[:3]

    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[int(index)]

        print(
            f"{rank}. {chunk['chunk_id']} "
            f"| BM25 score: {scores[index]:.4f}"
        )