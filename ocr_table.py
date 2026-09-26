from pathlib import Path
import subprocess
import pymupdf

project_dir = Path(__file__).parent
pdf_path = project_dir / "data" / "raw" / "pgcet.pdf"
tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

image_path = project_dir / "data" / "images" / "bagalkote_table.png"
output_base = project_dir / "data" / "extracted" / "bagalkote_table"

image_path.parent.mkdir(parents=True, exist_ok=True)
output_base.parent.mkdir(parents=True, exist_ok=True)

with pymupdf.open(pdf_path) as document:
    page = document[2]  # PDF page 3
    width = page.rect.width
    height = page.rect.height

    # Approximate bounds based on the uploaded page.
    crop = pymupdf.Rect(
        0,
        height * 0.265,
        width,
        height * 0.425,
    )

    image = page.get_pixmap(clip=crop, dpi=300)
    image.save(image_path)

subprocess.run(
    [
        str(tesseract),
        str(image_path),
        str(output_base),
        "-l", "eng",
        "--psm", "6",
        "-c", "preserve_interword_spaces=1",
    ],
    check=True,
)

print(f"Crop saved: {image_path}")
print(f"OCR saved: {output_base.with_suffix('.txt')}")