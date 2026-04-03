import os
from datasets import load_dataset
from tqdm import tqdm

DATASET_ID = "tanganke/stanford_cars"
OUTPUT_DIR = "data/1_catalog_images"


def setup_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_and_format_dataset():
    print(f"Connecting to Hugging Face to download: {DATASET_ID}...")

    dataset_train = load_dataset(DATASET_ID, split="train")
    dataset_test = load_dataset(DATASET_ID, split="test")

    class_names = dataset_train.features['label'].names
    setup_directory()

    def process_split(split_data, prefix):
        for index, item in enumerate(tqdm(split_data, desc=f"Extracting {prefix} data")):
            image = item['image']
            label_id = item['label']

            raw_class_name = class_names[label_id]
            formatted_class_name = raw_class_name.lower().replace(' ', '_').replace('/', '_')

            class_dir = os.path.join(OUTPUT_DIR, formatted_class_name)
            os.makedirs(class_dir, exist_ok=True)

            if image.mode != 'RGB':
                image = image.convert('RGB')

            image_filename = f"{prefix}_{index:05d}.jpg"
            image_path = os.path.join(class_dir, image_filename)
            image.save(image_path)

    process_split(dataset_train, "train")
    process_split(dataset_test, "test")

    print(f"\nSuccess! All catalog images have been extracted to: {OUTPUT_DIR}")


if __name__ == "__main__":
    download_and_format_dataset()