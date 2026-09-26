from pathlib import Path
import argparse
import pymupdf

parser = argparse.ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()

project_dir = Path(__file__).parent
pdf_path = project_dir / "data" / "raw" / args.filename
output_dir = project_dir / "data" / "extracted"
output_dir.mkdir(parents=True, exist_ok=True)

with pymupdf.open(pdf_path) as document:
    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        if not text.strip():
            print(f"Page {page_number}: no text extracted; may need OCR.")
            continue

        output_path = (
            output_dir / f"{pdf_path.stem}_page_{page_number}.txt"
        )
        output_path.write_text(text, encoding="utf-8")
        print(f"Saved page {page_number}: {len(text)} characters")