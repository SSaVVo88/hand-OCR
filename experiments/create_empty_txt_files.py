import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGES_DIR = os.path.join(BASE_DIR, "data", "lines", "images")
TXT_DIR = os.path.join(BASE_DIR, "data", "lines", "txt")

os.makedirs(TXT_DIR, exist_ok=True)

created = 0

for root, _, files in os.walk(IMAGES_DIR):
    folder_name = os.path.basename(root)

    for file in files:
        if file.lower().endswith(".png"):
            line_name = os.path.splitext(file)[0]
            txt_name = f"{folder_name}_{line_name}.txt"
            txt_path = os.path.join(TXT_DIR, txt_name)

            if not os.path.exists(txt_path):
                with open(txt_path, "w", encoding="utf-8"):
                    pass
                created += 1

print(f"Utworzono {created} plików TXT w data/lines/txt")

