from pathlib import Path
import subprocess

project_dir = Path(__file__).parent

tesseract_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
# image_path = project_dir / "data" / "images" / "page_1.png"
# output_base = project_dir / "data" / "extracted" / "page_1_ocr"
image_path = project_dir / "data" / "images" / "notification_25092026.png"
output_base = project_dir / "data" / "extracted" / "notification_25092026"

output_base.parent.mkdir(parents=True, exist_ok=True)

subprocess.run(
    [
        str(tesseract_path),
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
)

print(f"OCR text saved to: {output_base.with_suffix('.txt')}")