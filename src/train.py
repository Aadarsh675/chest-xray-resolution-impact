# train_detr_simplified.py
# This script fine-tunes a DETR model on a custom COCO-format dataset using Hugging Face Transformers.
# Dataset structure:
# data/
# ├── annotations/
# │   ├── train_annotations_coco.json
# │   └── test_annotations_coco.json  # For validation
# └── images/
#     ├── train/
#     └── test/  # For validation

import os
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import Trainer, TrainingArguments

# Specify data paths
DATA_DIR = "data"
ANNO_DIR = os.path.join(DATA_DIR, "annotations")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

TRAIN_ANNO_FILE = os.path.join(ANNO_DIR, "train_annotations_coco.json")
TRAIN_IMG_DIR = os.path.join(IMAGE_DIR, "train")

VAL_ANNO_FILE = os.path.join(ANNO_DIR, "test_annotations_coco.json")
VAL_IMG_DIR = os.path.join(IMAGE_DIR, "test")

# Custom Dataset class for COCO format
class CocoDetection(Dataset):
    def __init__(self, img_folder, anno_file, processor):
        self.coco = COCO(anno_file)
        self.img_folder = img_folder
        self.processor = processor
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Prepare target for processor
        target = {
            'image_id': img_id,
            'annotations': anns
        }

        # Process image and annotations
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        return encoding

def main():
    # Load image processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Load categories from train annotations to get num_labels and id2label
    with open(TRAIN_ANNO_FILE, 'r') as f:
        coco_data = json.load(f)
    categories = coco_data['categories']
    id2label = {cat['id']: cat['name'] for cat in categories}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    # Print dataset-specific info
    print(f"Dataset specifics: Medical chest X-ray images with {num_labels} disease classes: {list(id2label.values())}")

    # Load model with custom num_labels
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True  # To handle class mismatch from pretrained
    )

    # Create datasets
    train_dataset = CocoDetection(TRAIN_IMG_DIR, TRAIN_ANNO_FILE, processor)
    val_dataset = CocoDetection(VAL_IMG_DIR, VAL_ANNO_FILE, processor)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./detr_finetuned",
        num_train_epochs=50,
        per_device_train_batch_size=2,  # Small batch due to memory
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False  # Important for custom datasets
    )

    # Custom collate function for batching
    def collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        pixel_mask = torch.stack([item['pixel_mask'] for item in batch])
        labels = [item['labels'] for item in batch]
        return {'pixel_values': pixel_values, 'pixel_mask': pixel_mask, 'labels': labels}

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=processor  # Processor acts as tokenizer for images
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model("./detr_finetuned_model")

if __name__ == "__main__":
    main()
