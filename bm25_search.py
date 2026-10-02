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

question = "When can M.Tech candidates enter the examination hall?"

query_tokens = tokenize(question)

scores = bm25.get_scores(query_tokens)

top_indices = np.argsort(scores)[::-1][:3]

print(f"\nQuestion: {question}")

for rank, index in enumerate(top_indices, start=1):
    chunk = chunks[int(index)]

    print(
        f"{rank}. {chunk['chunk_id']} "
        f"| BM25 score: {scores[index]:.4f}"
    )