import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

model = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    device="cpu",
)

question = "When can M.Tech candidates enter the examination hall?"

# Convert the question into a 384-dimensional vector.
query_vector = model.encode(
    "query: " + question,
    normalize_embeddings=True,
)

vector_string = "[" + ",".join(map(str, query_vector.tolist())) + "]"

with psycopg.connect(
    host="127.0.0.1",
    port=5433,
    dbname=os.environ["POSTGRES_DB"],
    user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"],
) as conn:

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                chunk_id,
                content,
                1 - (embedding <=> %s::vector) AS similarity
            FROM rag_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT 5;
            """,
            (vector_string, vector_string),
        )

        results = cur.fetchall()
# Compare pgvector with our original NumPy retrieval
import json
import numpy as np

chunks = json.loads(
    (PROJECT_DIR / "data" / "processed" / "chunks_embedded.json").read_text(
        encoding="utf-8"
    )
)

vectors = np.asarray(
    [chunk["embedding"] for chunk in chunks],
    dtype=np.float32,
)

scores = vectors @ query_vector

print("\n--- NumPy Top 5 ---")

for index in np.argsort(scores)[::-1][:5]:
    print(chunks[int(index)]["chunk_id"], round(float(scores[index]), 4))

for chunk_id, content, similarity in results:
    print(f"\nChunk: {chunk_id}")
    print(f"Similarity: {similarity:.4f}")
    print(f"Preview: {content[:200]}")
    print("-" * 50)
