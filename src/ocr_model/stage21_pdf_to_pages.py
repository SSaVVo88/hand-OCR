"""
Stage 2.1.1
PDF → Page Images Extraction

This module converts multipage PDF files containing handwritten scans
into high-resolution page images suitable for further preprocessing
(line segmentation, normalization, etc.).

Input:
- PDF files stored in data/raw/

Output:
- PNG page images stored in data/pages/

Design goals:
- Preserve handwriting stroke quality
- Deterministic file naming
- Minimal preprocessing (no resizing beyond DPI control)
"""

from pathlib import Path
from typing import Union
from pdf2image import convert_from_path


def extract_pages(
    pdf_path: Union[str, Path],
    output_dir: Union[str, Path],
    dpi: int = 300,
    image_format: str = "png",
) -> None:
    """
    Extract pages from a PDF file and save them as images.

    Parameters
    ----------
    pdf_path : str or Path
        Path to the input PDF file.
    output_dir : str or Path
        Directory where page images will be saved.
    dpi : int, optional
        Resolution for PDF rendering. Default is 300 DPI.
    image_format : str, optional
        Output image format ("png" recommended). Default is "png".
    """

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pages = convert_from_path(
        pdf_path,
        dpi=dpi,
    )

    pdf_stem = pdf_path.stem

    for idx, page in enumerate(pages, start=1):
        page_name = f"{pdf_stem}_page_{idx:03d}.{image_format}"
        page_path = output_dir / page_name
        page.save(page_path)

    print(
        f"[INFO] Extracted {len(pages)} pages from '{pdf_path.name}' "
        f"to '{output_dir}' at {dpi} DPI."
    )


if __name__ == "__main__":
    # Example manual run for testing/debugging

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    if not RAW_DATA_DIR.exists():
        raise RuntimeError(f"RAW_DATA_DIR does not exist: {RAW_DATA_DIR}")
    PAGES_DIR = PROJECT_ROOT / "data" / "pages"
    print("[DEBUG] RAW_DATA_DIR =", RAW_DATA_DIR)
    print("[DEBUG] PDFs found:", list(RAW_DATA_DIR.glob("*.pdf")))
    for pdf_file in RAW_DATA_DIR.glob("*.pdf"):
        extract_pages(
            pdf_path=pdf_file,
            output_dir=PAGES_DIR,
            dpi=300,
        )