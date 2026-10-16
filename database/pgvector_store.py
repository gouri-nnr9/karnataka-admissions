import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_DIR / ".env")


def search_pgvector(query_vector, limit=10):
    """Retrieve the most similar chunks from PostgreSQL."""

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
                    metadata,
                    embedding_model,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM rag_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vector_string, vector_string, limit),
            )

            rows = cur.fetchall()

    return [
        {
            "chunk_id": row[0],
            "text": row[1],
            "metadata": row[2],
            "embedding_model": row[3],
            "similarity": row[4],
        }
        for row in rows
    ]
