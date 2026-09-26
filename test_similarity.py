from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer

project_dir = Path(__file__).parent
file_path = (
    project_dir / "data" / "processed"
    / "notification_25092026_embedded.json"
)

notice = json.loads(file_path.read_text(encoding="utf-8"))

model = SentenceTransformer(notice["embedding_model"], device="cpu")
document_vector = np.array(notice["embedding"], dtype=np.float32)

questions = [
    "When were the PGCET 2026 first-round results published?",
    "When will the choice entry and fee payment schedule be published?",
    "How do I bake a chocolate cake?",
]

for question in questions:
    query_vector = model.encode(
        "query: " + question,
        normalize_embeddings=True,
    )

    similarity = float(np.dot(query_vector, document_vector))

    print(f"\nQuestion: {question}")
    print(f"Similarity: {similarity:.4f}")