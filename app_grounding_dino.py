# app_grounding_dino.py

import torch
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-base"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Grounding DINO on: {device}")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
model.eval()


def draw_boxes(image, boxes, scores, labels):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for box, score, label in zip(boxes, scores, labels):
        x_min, y_min, x_max, y_max = [int(v) for v in box]

        draw.rectangle(
            [x_min, y_min, x_max, y_max],
            outline="red",
            width=4,
        )

        text = f"{label}: {score:.2f}"

        text_bbox = draw.textbbox((x_min, y_min), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        draw.rectangle(
            [x_min, y_min - text_h - 6, x_min + text_w + 8, y_min],
            fill="red",
        )
        draw.text(
            (x_min + 4, y_min - text_h - 4),
            text,
            fill="white",
            font=font,
        )

    return image


def detect_objects(image, prompt, box_threshold, text_threshold):
    if image is None:
        return None, "Wrzuć obraz."

    if not prompt.strip():
        prompt = "car ."

    if not prompt.strip().endswith("."):
        prompt = prompt.strip() + " ."

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    boxes = results["boxes"].cpu().tolist()
    scores = results["scores"].cpu().tolist()
    labels = results["labels"]

    output_image = draw_boxes(image.copy(), boxes, scores, labels)

    info = f"Znaleziono obiektów: {len(boxes)}"

    if boxes:
        info += "\n\n"
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels), start=1):
            rounded_box = [round(v, 2) for v in box]
            info += f"{i}. {label} | score={score:.3f} | box={rounded_box}\n"

    return output_image, info


with gr.Blocks(title="Grounding DINO Bounding Box Demo") as demo:
    gr.Markdown(
        """
        # Grounding DINO — wykrywanie obiektów

        Wrzuć zdjęcie, wpisz prompt, np.:

        - `car .`
        - `vehicle .`
        - `truck .`
        - `license plate .`
        - `red car .`
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil",
                label="Zdjęcie wejściowe",
            )
            prompt = gr.Textbox(
                value="car .",
                label="Prompt",
                placeholder="np. car .",
            )
            box_threshold = gr.Slider(
                minimum=0.05,
                maximum=0.9,
                value=BOX_THRESHOLD,
                step=0.05,
                label="Box threshold",
            )
            text_threshold = gr.Slider(
                minimum=0.05,
                maximum=0.9,
                value=TEXT_THRESHOLD,
                step=0.05,
                label="Text threshold",
            )
            run_button = gr.Button("Wykryj bounding boxy")

        with gr.Column():
            output_image = gr.Image(
                type="pil",
                label="Zdjęcie z bounding boxami",
            )
            output_info = gr.Textbox(
                label="Wyniki",
                lines=10,
            )

    run_button.click(
        fn=detect_objects,
        inputs=[input_image, prompt, box_threshold, text_threshold],
        outputs=[output_image, output_info],
    )

if __name__ == "__main__":
    demo.launch()