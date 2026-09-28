from pathlib import Path
import pymupdf

project_dir = Path(__file__).parent
pdf_path = project_dir / "data" / "raw" / "pgcet.pdf"

output_dir = project_dir / "data" / "images"
output_dir.mkdir(parents=True, exist_ok=True)

with pymupdf.open(pdf_path) as document:
    page = document[105]
    image = page.get_pixmap(dpi=300)
    output_path = output_dir / "pgcet_page_106.png"
    image.save(output_path)

print(f"Saved: {output_path}")