from pathlib import Path
import json
import re
import os

import numpy as np
from dotenv import load_dotenv
from google import genai
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


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
# Models
# --------------------------------------------------

model = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    device="cpu",
)

reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    device="cpu",
    max_length=1024,
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

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        kannada_query = response.text.strip()

        return [
            question,
            kannada_query,
        ]

    except Exception as error:
        print(f"Query expansion failed: {error}")
        print("Falling back to original English query.")

        return [question]


# --------------------------------------------------
# Search one query
# --------------------------------------------------

def retrieve_rankings(search_query, top_k=50):

    # Semantic
    query_vector = model.encode(
        "query: " + search_query,
        normalize_embeddings=True,
    )

    semantic_scores = document_vectors @ query_vector

    semantic_ranking = np.argsort(
        semantic_scores
    )[::-1][:top_k]

    # BM25
    query_tokens = tokenize(search_query)

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    bm25_ranking = np.argsort(
        bm25_scores
    )[::-1][:top_k]

    return semantic_ranking, bm25_ranking


# --------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------

def reciprocal_rank_fusion(rankings, k=60):

    rrf_scores = {}

    for ranking in rankings:

        for rank, index in enumerate(
            ranking,
            start=1,
        ):

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
# Evaluation counters
# --------------------------------------------------

answerable_count = 0

hit_at_1 = 0
hit_at_3 = 0


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

for case in test_cases:

    question = case["question"]

    expanded_queries = expand_query(question)

    # English + Kannada
    bilingual_query = " ".join(
        expanded_queries
    )

    print("\n--------------------------------")
    print(f"Question: {question}")
    print(f"Bilingual query: {bilingual_query}")


    # --------------------------------------------------
    # Path 1: Original English query
    # --------------------------------------------------

    english_semantic, english_bm25 = (
        retrieve_rankings(question)
    )


    # --------------------------------------------------
    # Path 2: Bilingual query
    # --------------------------------------------------

    bilingual_semantic, bilingual_bm25 = (
        retrieve_rankings(bilingual_query)
    )


    # --------------------------------------------------
    # Combine all retrieval paths
    # --------------------------------------------------

    rankings = [
        english_semantic,
        english_bm25,
        bilingual_semantic,
        bilingual_bm25,
    ]

    hybrid_ranking = reciprocal_rank_fusion(
        rankings
    )

        # --------------------------------------------------
    # Build candidate union
    # --------------------------------------------------

    candidate_indices = []

    candidate_sources = [
        english_semantic[:10],
        english_bm25[:10],
        bilingual_semantic[:10],
        bilingual_bm25[:10],
    ]

    for source in candidate_sources:
        for index in source:
            index = int(index)

            if index not in candidate_indices:
                candidate_indices.append(index)

    print(
        f"Candidate pool size: {len(candidate_indices)}"
    )

    candidate_pairs = [
        (
            question,
            chunks[index]["text"],
        )
        for index in candidate_indices
    ]

    # --------------------------------------------------
    # Rerank candidates
    # --------------------------------------------------

    reranker_scores = reranker.predict(
        candidate_pairs
    )

    reranked_results = sorted(
        zip(
            candidate_indices,
            reranker_scores,
        ),
        key=lambda item: item[1],
        reverse=True,
    )


    # --------------------------------------------------
    # Show reranked top 3
    # --------------------------------------------------

    print("Reranked Top 3:")

    for rank, (index, score) in enumerate(
        reranked_results[:3],
        start=1,
    ):

        print(
            f"{rank}. "
            f"{chunks[index]['chunk_id']} "
            f"| score: {float(score):.4f}"
        )


    # --------------------------------------------------
    # Evaluate answerable questions
    # --------------------------------------------------

    if case["answerable"]:

        answerable_count += 1

        expected_ids = set(
            case["expected_chunk_ids"]
        )

        reranked_ids = [
            chunks[index]["chunk_id"]
            for index, score in reranked_results
        ]

        expected_ranks = [
            rank
            for rank, chunk_id in enumerate(
                reranked_ids,
                start=1,
            )
            if chunk_id in expected_ids
        ]

        print(
            f"Expected chunk RERANKED rank: "
            f"{expected_ranks}"
        )

        if any(
            chunk_id in expected_ids
            for chunk_id in reranked_ids[:1]
        ):
            hit_at_1 += 1

        if any(
            chunk_id in expected_ids
            for chunk_id in reranked_ids[:3]
        ):
            hit_at_3 += 1


# --------------------------------------------------
# Final evaluation
# --------------------------------------------------

print("\n================================")
print("FINAL RESULTS")
print("================================")

print(
    f"Reranker Hit@1: "
    f"{hit_at_1}/{answerable_count} "
    f"({hit_at_1 / answerable_count * 100:.0f}%)"
)

print(
    f"Reranker Hit@3: "
    f"{hit_at_3}/{answerable_count} "
    f"({hit_at_3 / answerable_count * 100:.0f}%)"
)