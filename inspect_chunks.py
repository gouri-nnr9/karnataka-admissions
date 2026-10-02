from pathlib import Path
import json

project_dir = Path(__file__).parent

chunks = json.loads(
    (project_dir / "data" / "processed" / "chunks.json").read_text(
        encoding="utf-8"
    )
)

sources = [
    "Bell_Timings_PGCET_2026_english_Mtechenglish.pdf",
    "pgcet_notification_25092026english.pdf",
]

for source in sources:
    print(f"\n\n===== {source} =====")

    for chunk in chunks:
        if chunk["metadata"]["source"] == source:
            print(f"\n--- {chunk['chunk_id']} ---")
            print(chunk["text"])