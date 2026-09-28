from pathlib import Path
import json
from sentence_transformers import SentenceTransformer

project_dir = Path(__file__).parent
input_path = project_dir / "data" / "processed" / "chunks.json"

chunks = json.loads(input_path.read_text(encoding="utf-8"))

model_name = "intfloat/multilingual-e5-small"
model = SentenceTransformer(model_name, device="cpu")

inputs = ["passage: " + chunk["text"] for chunk in chunks]

# Validate every chunk before embedding.
for chunk, text in zip(chunks, inputs):
    token_count = len(
        model.tokenizer(text, truncation=False)["input_ids"]
    )
    print(f"{chunk['chunk_id']}: {token_count} tokens")

    if token_count > model.max_seq_length:
        raise ValueError(
            f"{chunk['chunk_id']} exceeds {model.max_seq_length} tokens. "
            "Split it before embedding."
        )

embeddings = model.encode(
    inputs,
    normalize_embeddings=True,
    batch_size=8,
)

for chunk, embedding in zip(chunks, embeddings):
    chunk["embedding_model"] = model_name
    chunk["embedding"] = embedding.tolist()

output_path = input_path.with_name("chunks_embedded.json")
output_path.write_text(
    json.dumps(chunks, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"\nSaved {len(chunks)} embeddings to {output_path.name}")