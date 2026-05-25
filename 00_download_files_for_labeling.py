import os
import shutil

CROPS_DIR = os.path.join("data", "4_debug_crops")
OUTPUT_DIR = os.path.join("data", "pre_label_images")

PRELABEL_FILE = "images_for_labeling.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(PRELABEL_FILE) as f:
    for line in f:

        name = line.strip()

        if not name:
            continue

        src_path = os.path.join(CROPS_DIR, name)
        dst_path = os.path.join(OUTPUT_DIR, name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            print(f"Skopiowano: {name}")
        else:
            print(f"BRAK PLIKU: Skrypt szukał go pod ścieżką -> {src_path}")