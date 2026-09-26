from pathlib import Path
import json
import re

project_dir = Path(__file__).parent
filename = "Bell_Timings_PGCET_2026_english_Mtechenglish"

text_path = project_dir / "data" / "extracted" / f"{filename}_page_1.txt"
text = text_path.read_text(encoding="utf-8")

# Find the beginning of each numbered bell row.
pattern = (
    r"(?m)^\s*[1-7]\s*\n\s*"
    r"(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH)"
    r"\b"
)
matches = list(re.finditer(pattern, text))

if len(matches) != 7:
    raise ValueError(
        f"Expected 7 bell sections, found {len(matches)}. "
        "Check the extracted text before continuing."
    )

# Separate the rules after the seventh bell.
rules_start = text.find("Only Non-Programmable Calculators")

if rules_start < matches[-1].start():
    raise ValueError("Could not locate the final rules section.")

sections = [("overview", text[:matches[0].start()])]

for index, match in enumerate(matches):
    end = (
        matches[index + 1].start()
        if index + 1 < len(matches)
        else rules_start
    )
    sections.append((f"bell_{index + 1}", text[match.start():end]))

sections.append(("general_rules", text[rules_start:]))

chunks = []

for section_name, section_text in sections:
    # Remove repeated whitespace without changing the words.
    cleaned_text = " ".join(section_text.split())

    chunks.append({
        "chunk_id": f"pgcet_2026_mtech_{section_name}",
        "text": (
            "PGCET 2026 M.E./M.Tech examination instructions. "
            + cleaned_text
        ),
        "metadata": {
            "exam": "PGCET",
            "admission_year": 2026,
            "courses": ["M.E.", "M.Tech"],
            "language": "en",
            "section": section_name,
            "source_file": f"{filename}.pdf",
            "source_url": (
                "https://cetonline.karnataka.gov.in/"
                f"keawebentry456/pgcet2026/{filename}.pdf"
            ),
            "page": 1,
            "extraction_method": "pymupdf",
        },
    })

output_path = project_dir / "data" / "processed" / "bell_timings_chunks.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(
    json.dumps(chunks, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Saved {len(chunks)} chunks")
for chunk in chunks:
    print(chunk["chunk_id"], "-", len(chunk["text"]), "characters")