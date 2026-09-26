from pathlib import Path
import json

project_dir = Path(__file__).parent

text_path = (
    project_dir / "data" / "extracted"
    / "notification_25092026_reviewed.txt"
)

text = text_path.read_text(encoding="utf-8").strip()

if not text:
    raise ValueError("The notice text file is empty.")

notice = {
    "chunk_id": "pgcet_2026_25092026_page1_chunk1",
    "text": text,
    "metadata": {
        "exam": "PGCET",
        "admission_year": 2026,
        "courses": ["MBA", "MCA", "M.Tech", "M.Arch"],
        "notice_date": "2026-09-25",
        "language": "kn",
        "source_file": "pgcet_notification_25092026english.pdf",
        "source_url": None,
        "page": 1,
        "extraction_method": "tesseract",
        "manually_reviewed": True,
    },
}

output_dir = project_dir / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "notification_25092026.json"

output_path.write_text(
    json.dumps(notice, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Saved: {output_path}")