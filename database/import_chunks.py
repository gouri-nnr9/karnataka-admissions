import json
import os
import sys
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb
from dotenv import load_dotenv

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "chunks_embedded.json"

load_dotenv(PROJECT_DIR / ".env")

# Load existing embeddings
chunks = json.loads(DATA_FILE.read_text(encoding="utf-8"))

print(f"Found {len(chunks)} chunks.")

# Validate before importing
for chunk in chunks:
    if len(chunk["embedding"]) != 384:
        raise ValueError(f"Invalid embedding dimension: {chunk['chunk_id']}")

# Connect to PostgreSQL
with psycopg.connect(
    host="127.0.0.1",
    port=5433,
    dbname=os.environ["POSTGRES_DB"],
    user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"],
) as conn:

    with conn.cursor() as cur:

        for chunk in chunks:
            cur.execute(
                """
                INSERT INTO rag_chunks (
                    chunk_id,
                    content,
                    metadata,
                    embedding_model,
                    embedding
                )
                VALUES (%s, %s, %s, %s, %s::vector)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding = EXCLUDED.embedding
                """,
                (
                    chunk["chunk_id"],
                    chunk["text"],
                    Jsonb(chunk["metadata"]),
                    chunk["embedding_model"],
                    json.dumps(chunk["embedding"]),
                ),
            )

        cur.execute("SELECT COUNT(*) FROM rag_chunks")
        total = cur.fetchone()[0]

    print(f"Database now contains {total} chunks.")

print("Import completed successfully!")
