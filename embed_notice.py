from pathlib import Path
import json
from sentence_transformers import SentenceTransformer

project_dir = Path(__file__).parent
input_path = (
    project_dir / "data" / "processed"
    / "notification_25092026.json"
)

notice = json.loads(input_path.read_text(encoding="utf-8"))

model_name = "intfloat/multilingual-e5-small"
print("Loading the model; the first run downloads its files...")
model = SentenceTransformer(model_name, device="cpu")

# E5 expects document text to begin with "passage: ".
embedding_input = "passage: " + notice["text"]

# Check length so the model doesn't silently discard text.
token_ids = model.tokenizer(
    embedding_input,
    truncation=False,
)["input_ids"]

print(f"Token count: {len(token_ids)}")
print(f"Model input limit: {model.max_seq_length}")

if len(token_ids) > model.max_seq_length:
    raise ValueError("This notice needs smaller chunks before embedding.")

embedding = model.encode(
    embedding_input,
    normalize_embeddings=True,
)

notice["embedding_model"] = model_name
notice["embedding"] = embedding.tolist()

output_path = input_path.with_name("notification_25092026_embedded.json")
output_path.write_text(
    json.dumps(notice, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Embedding dimensions: {len(embedding)}")
print(f"First five values: {embedding[:5]}")
print(f"Saved: {output_path}")