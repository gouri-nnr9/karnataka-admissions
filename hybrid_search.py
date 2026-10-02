from pathlib import Path
import json
import re
import os

import numpy as np
from dotenv import load_dotenv
from google import genai
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Paths and environment
# --------------------------------------------------

project_dir = Path(__file__).parent
processed_dir = project_dir / "data" / "processed"

load_dotenv(project_dir / ".env")

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

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

print(f"Loaded {len(chunks)} chunks.")


# --------------------------------------------------
# BM25
# --------------------------------------------------

def tokenize(text):
    return re.findall(r"\w+", text.lower())


tokenized_chunks = [
    tokenize(chunk["text"])
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


# --------------------------------------------------
# Semantic search
# --------------------------------------------------

model = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    device="cpu",
)

document_vectors = np.array(
    [chunk["embedding"] for chunk in chunks],
    dtype=np.float32,
)


# --------------------------------------------------
# Multilingual query expansion
# --------------------------------------------------

def expand_query(question):
    prompt = f"""
Translate the following search query into Kannada.

Keep names, exam names, dates, course names, and official terms accurate.
Return ONLY the Kannada translation. No explanation.

Query:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    kannada_query = response.text.strip()

    return [
        question,
        kannada_query,
    ]


# --------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------

def reciprocal_rank_fusion(rankings, k=60):
    rrf_scores = {}

    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            index = int(index)

            rrf_scores[index] = (
                rrf_scores.get(index, 0)
                + 1 / (k + rank)
            )

    return sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True,
    )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

for case in test_cases:

    question = case["question"]

    # English + Kannada
    expanded_queries = expand_query(question)

    print(f"\nQuestion: {question}")
    print(f"Queries: {expanded_queries}")

    rankings = []

    # Search using BOTH queries
    for search_query in expanded_queries:

        # Semantic search
        query_vector = model.encode(
            "query: " + search_query,
            normalize_embeddings=True,
        )

        semantic_scores = document_vectors @ query_vector

        semantic_ranking = np.argsort(
            semantic_scores
        )[::-1]

        rankings.append(
            semantic_ranking[:50]
        )

        # BM25 search
        query_tokens = tokenize(search_query)

        bm25_scores = bm25.get_scores(
            query_tokens
        )

        bm25_ranking = np.argsort(
            bm25_scores
        )[::-1]

        rankings.append(
            bm25_ranking[:50]
        )

    # English semantic
    # English BM25
    # Kannada semantic
    # Kannada BM25
    #          ↓
    #         RRF

    hybrid_ranking = reciprocal_rank_fusion(
        rankings
    )

    # --------------------------------------------------
    # Evaluation: find expected chunk
    # --------------------------------------------------

    if case["answerable"]:

        expected_ids = set(
            case["expected_chunk_ids"]
        )

        expected_ranks = []

        for rank, index in enumerate(
            hybrid_ranking,
            start=1,
        ):

            chunk_id = chunks[index]["chunk_id"]

            if chunk_id in expected_ids:
                expected_ranks.append(rank)

        print(
            f"Expected chunk HYBRID rank: "
            f"{expected_ranks}"
        )

    # --------------------------------------------------
    # Show top 3
    # --------------------------------------------------

    print("Top 3:")

    for rank, index in enumerate(
        hybrid_ranking[:3],
        start=1,
    ):

        print(
            f"{rank}. "
            f"{chunks[index]['chunk_id']}"
        )