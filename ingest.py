from pathlib import Path
import pymupdf
import subprocess
import tempfile
import json

PROJECT_DIR = Path(__file__).parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
EXTRACTED_DIR = PROJECT_DIR / "data" / "extracted"
TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def find_pdfs():
    return sorted(RAW_DIR.glob("*.pdf"))

def has_useful_text(text, min_chars=50):
    return len(text.strip()) >= min_chars

def ocr_page(page):
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / "page.png"
        output_base = Path(temp_dir) / "ocr_output"

        # Render PDF page as an image
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
        pix.save(image_path)

        # Run Tesseract
        subprocess.run(
            [
                str(TESSERACT_PATH),
                str(image_path),
                str(output_base),
                "-l",
                "kan+eng",
                "--psm",
                "6",
                "-c",
                "preserve_interword_spaces=1",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        text_path = output_base.with_suffix(".txt")
        return text_path.read_text(encoding="utf-8").strip()


def extract_pdf(pdf_path):
    pages = []
    

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text()
            extraction_method = "pymupdf"

            if not has_useful_text(text):
                 print(f"  Page {page_number}: insufficient text — running OCR...")
                 text = ocr_page(page)
                 extraction_method = "ocr"
                 if not has_useful_text(text):
                     print(f"  Page {page_number}: OCR did not produce enough text.")
                     continue
                 print(f"  Page {page_number}: OCR successful.")
                 #print(f"  OCR preview: {text[:200].replace(chr(10), ' ')}")

            pages.append({
                "source": pdf_path.name,
                "page": page_number,
                "text": text.strip(),
                "extraction_method": extraction_method
            })

    return pages


if __name__ == "__main__":
    pdfs = find_pdfs()
    all_pages = []

    print(f"Found {len(pdfs)} PDFs.\n")

    for pdf in pdfs:
        print(f"Processing: {pdf.name}")

        pages = extract_pdf(pdf)
        all_pages.extend(pages)

        print(f"  Extracted {len(pages)} text pages.\n")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / "documents.json"

    output_path.write_text(
        json.dumps(all_pages, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Saved {len(all_pages)} pages to:")
    print(output_path)