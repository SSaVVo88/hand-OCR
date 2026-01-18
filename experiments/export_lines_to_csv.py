from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LINES_ROOT = PROJECT_ROOT / "data" / "lines"
OUTPUT_CSV = LINES_ROOT / "lines.csv"

rows = []

for img_path in sorted(LINES_ROOT.rglob("line_*.png")):
    page_name = img_path.parent.name      # Adobe Scan ..._preproc
    line_name = img_path.stem              # line_001

    sample_id = f"{page_name}_{line_name}"

    rows.append({
        "sample_id": sample_id,
        "transcription": ""
    })

print(f"Znaleziono {len(rows)} plików PNG")



LINES_ROOT.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["sample_id", "transcription"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ CSV zapisany w: {OUTPUT_CSV.resolve()}")
