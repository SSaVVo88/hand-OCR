import os
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TXT_DIR = os.path.join(BASE_DIR, "data", "lines", "txt")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "lines", "lines.csv")

rows = []

for txt_file in sorted(os.listdir(TXT_DIR)):
    if txt_file.lower().endswith(".txt"):
        txt_path = os.path.join(TXT_DIR, txt_file)

        with open(txt_path, "r", encoding="utf-8") as f:
            transcription = f.read().strip()

        image_name = txt_file.replace(".txt", ".png")

        rows.append([image_name, transcription])

with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(
        csvfile,
        delimiter=";",
        quotechar='"',
        quoting=csv.QUOTE_ALL
    )

    writer.writerow(["image_name", "transcription"])
    writer.writerows(rows)

print(f"✅ CSV zapisany: {OUTPUT_CSV} ({len(rows)} wierszy)")
