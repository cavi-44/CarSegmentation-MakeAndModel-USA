import os
import json
import csv
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# --- KONFIGURACJA ŚCIEŻEK ---
IMAGES_DIR = "data/raw_road_traffic/train/images"
LABELS_DIR = "data/raw_road_traffic/train/labels"
CATALOG_DIR = "data/1_catalog_images"
OUTPUT_FILE = "data/train_autolabeled.jsonl"
LOG_FILE = "data/processing_logs.csv"
CROPS_DIR = "data/4_debug_crops"

os.makedirs(CROPS_DIR, exist_ok=True)

# --- PROGI I PARAMETRY --- || "<- [wartosc przed update'em]"
MIN_BBOX_AREA = 1500 # <- 3000
VEHICLE_CLASSES = ["8", "11"]
CROP_PADDING = 0.0 # <- 0.10

# Progi oceny
THRESHOLD_GOOD = 0.40
THRESHOLD_MEDIUM = 0.20
THRESHOLD_BAD = 0.1 # <- 0.12



def get_rating(prob):
    if prob >= THRESHOLD_GOOD: return "DOBRA"
    if prob >= THRESHOLD_MEDIUM: return "SREDNIA"
    if prob >= THRESHOLD_BAD: return "SLABA"
    return "IGNORUJ"


def main():
    print("Inicjalizacja Nauczyciela (CLIP) z pełną analityką Top-5...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    class_names = [d.replace('_', ' ') for d in os.listdir(CATALOG_DIR) if os.path.isdir(os.path.join(CATALOG_DIR, d))]
    text_prompts = [f"a photo of a {car}" for car in class_names]

    jsonl_data = []
    processed_images = 0
    labeled_cars = 0

    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))]

    with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as log_f:
        log_writer = csv.writer(log_f)

        # Nowe, szerokie nagłówki dla pełnej analityki
        headers = [
            "image_name", "crop_id", "bbox_coords", "area_px", "status",
            "p1_class", "p1_conf", "p1_rating",
            "p2_class", "p2_conf", "p2_rating",
            "p3_class", "p3_conf", "p3_rating",
            "p4_class", "p4_conf", "p4_rating",
            "p5_class", "p5_conf", "p5_rating"
        ]
        log_writer.writerow(headers)

        for img_name in tqdm(image_files):
            base_name = os.path.splitext(img_name)[0]
            label_file = os.path.join(LABELS_DIR, f"{base_name}.txt")
            img_path = os.path.join(IMAGES_DIR, img_name)

            if not os.path.exists(label_file): continue

            try:
                image = Image.open(img_path).convert("RGB")
                img_w, img_h = image.size
            except:
                continue

            image_annotations = {"file_name": img_path, "text": "", "bboxes": [], "labels": []}
            bbox_idx = 0

            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts or parts[0] not in VEHICLE_CLASSES: continue

                    bbox_idx += 1
                    _, xc, yc, w, h = map(float, parts)

                    w_px, h_px = w * img_w, h * img_h
                    x_min_raw = (xc - w / 2) * img_w
                    y_min_raw = (yc - h / 2) * img_h

                    pad_w, pad_h = w_px * CROP_PADDING, h_px * CROP_PADDING

                    # Pilnujemy, by krawędzie nie wyszły poza zdjęcie
                    x_min = int(max(0, x_min_raw - pad_w))
                    y_min = int(max(0, y_min_raw - pad_h))
                    x_max = int(min(img_w, x_min_raw + w_px + pad_w))
                    y_max = int(min(img_h, y_min_raw + h_px + pad_h))

                    bbox_area = int((x_max - x_min) * (y_max - y_min))
                    bbox_coords = f"[{x_min}, {y_min}, {x_max}, {y_max}]"

                    # Filtr 1: Zbyt małe auto (N/A dla wyników)
                    if bbox_area < MIN_BBOX_AREA:
                        log_row = [img_name, bbox_idx, bbox_coords, bbox_area, "REJECTED_SIZE"]
                        # Wypełniamy resztę kolumn pustymi wartościami
                        log_row.extend(["N/A"] * 15)
                        log_writer.writerow(log_row)
                        continue

                    # Wycinanie i analiza
                    crop = image.crop((x_min, y_min, x_max, y_max))
                    inputs = processor(text=text_prompts, images=crop, return_tensors="pt", padding=True).to(device)

                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = outputs.logits_per_image.softmax(dim=1)

                    top_probs, top_idxs = torch.topk(probs, k=min(5, len(class_names)), dim=1)

                    results = []
                    for i in range(top_probs.size(1)):
                        p = top_probs[0][i].item()
                        name = class_names[top_idxs[0][i].item()]
                        results.append({"name": name, "prob": p, "rating": get_rating(p)})

                    # Upewniamy się, że mamy dokładnie 5 wyników do CSV
                    while len(results) < 5:
                        results.append({"name": "N/A", "prob": 0.0, "rating": "N/A"})

                    top_1 = results[0]
                    status = "ACCEPTED" if top_1['rating'] != "IGNORUJ" else "REJECTED_CONF"

                    log_row = [img_name, bbox_idx, bbox_coords, bbox_area, status]
                    for res in results:
                        prob_str = f"{res['prob']:.4f}" if isinstance(res['prob'], float) else "N/A"
                        log_row.extend([res['name'], prob_str, res['rating']])

                    log_writer.writerow(log_row)

                    # Jeśli Top-1 jest akceptowalne, zapisujemy obrazek i do JSONL
                    if status == "ACCEPTED":
                        crop.save(os.path.join(CROPS_DIR, f"{base_name}_crop_{bbox_idx}.jpg"), quality=95)
                        image_annotations["bboxes"].append([float(x_min), float(y_min), float(x_max), float(y_max)])
                        image_annotations["labels"].append(top_1['name'])
                        labeled_cars += 1

            if image_annotations["bboxes"]:
                image_annotations["text"] = " . ".join(list(set(image_annotations["labels"]))) + " ."
                jsonl_data.append(image_annotations)
                processed_images += 1

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in jsonl_data: f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\nZakończono! Przetworzono {processed_images} zdjęć, oznaczono {labeled_cars} aut.")
    print(f"Logi ze szczegółowymi predykcjami zapisano w: {LOG_FILE}")


if __name__ == "__main__":
    main()