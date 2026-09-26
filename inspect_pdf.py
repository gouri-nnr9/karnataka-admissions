from pathlib import Path
from pypdf import PdfReader

project_dir = Path(__file__).parent
pdf_path = project_dir / "data" / "raw" / "pgcet.pdf"

output_dir = project_dir / "data" / "extracted"
output_dir.mkdir(parents=True, exist_ok=True)

reader = PdfReader(pdf_path)
print(f"Total pages: {len(reader.pages)}")

# These are PDF page positions, starting at 1.
page_numbers = [1, 3]

for page_number in page_numbers:
    page = reader.pages[page_number - 1]
    text = page.extract_text() or ""

    output_path = output_dir / f"page_{page_number}.txt"
    output_path.write_text(text, encoding="utf-8")

    print(f"Page {page_number}: saved {len(text)} characters")