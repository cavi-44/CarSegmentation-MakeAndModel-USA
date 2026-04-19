import os
import json
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_JSONL = "data/train_autolabeled.jsonl"
OUTPUT_DIR = "data/5_grounding_dino_eval"
MODEL_ID = "IDEA-Research/grounding-dino-base"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
IOU_THRESHOLD_FOR_TP = 0.5
MAX_DEBUG_IMAGES = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_iou(box1, box2):
    # box format: [x_min, y_min, x_max, y_max]
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Grounding DINO model on {device}...")
    
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

    # Load ground truth data
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} images for evaluation.")

    total_gt_boxes = 0
    total_pred_boxes = 0
    true_positives = 0
    sum_iou = 0.0
    matched_boxes_count = 0

    saved_debug_images = 0

    for idx, entry in enumerate(tqdm(data)):
        # Handle Windows paths just in case by replacing backslashes if needed, or keeping it
        img_path = entry["file_name"]
        gt_bboxes = entry["bboxes"]
        text_prompt = entry["text"]

        if not os.path.exists(img_path):
            continue
            
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            continue

        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)

        # target_sizes requires a list of tuples containing (height, width)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]] 
        )[0]
        
        pred_bboxes = results["boxes"].cpu().numpy().tolist()
        
        total_gt_boxes += len(gt_bboxes)
        total_pred_boxes += len(pred_bboxes)

        # Match predictions to ground truth
        matched_gt_indices = set()
        
        for pred_box in pred_bboxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_bboxes):
                if gt_idx in matched_gt_indices:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    
            if best_gt_idx != -1:
                matched_gt_indices.add(best_gt_idx)
                sum_iou += best_iou
                matched_boxes_count += 1
                if best_iou >= IOU_THRESHOLD_FOR_TP:
                    true_positives += 1

        # Draw debug images for a few examples
        if saved_debug_images < MAX_DEBUG_IMAGES:
            draw = ImageDraw.Draw(image)
            # GT in Green
            for box in gt_bboxes:
                draw.rectangle(tuple(box), outline="green", width=3)
            # Pred in Red
            for box in pred_bboxes:
                draw.rectangle(tuple(box), outline="red", width=3)
                
            base_name = os.path.basename(img_path)
            image.save(os.path.join(OUTPUT_DIR, f"eval_{base_name}"))
            saved_debug_images += 1

    precision = true_positives / total_pred_boxes if total_pred_boxes > 0 else 0
    recall = true_positives / total_gt_boxes if total_gt_boxes > 0 else 0
    average_iou = sum_iou / matched_boxes_count if matched_boxes_count > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- Evaluation Results ---")
    print(f"Total Images Evaluated: {len(data)}")
    print(f"Total Ground Truth Boxes: {total_gt_boxes}")
    print(f"Total Predicted Boxes: {total_pred_boxes}")
    print(f"True Positives (IoU >= {IOU_THRESHOLD_FOR_TP}): {true_positives}")
    print("--------------------------")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1_score:.4f}")
    print(f"Average IoU (matched boxes): {average_iou:.4f}")
    print(f"Debug images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
