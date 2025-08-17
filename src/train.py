# train_detr_simplified.py
# Fine-tunes DETR on NIH Chest X-ray dataset in COCO format.
# Dataset structure:
# data/
# ├── annotations/
# │ ├── train_annotations_coco.json
# │ └── test_annotations_coco.json
# └── images/
# ├── train/
# └── test/
import os
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
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
        image = Image.open(img_path).convert("RGB")  # Convert grayscale to RGB
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        target = {
            'image_id': img_id,
            'annotations': anns
        }
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        return encoding

def main():
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load processor and model
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Load categories
    with open(TRAIN_ANNO_FILE, 'r') as f:
        coco_data = json.load(f)
    categories = coco_data['categories']
    id2label = {cat['id']: cat['name'] for cat in categories}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)
    print(f"Dataset: NIH Chest X-ray with {num_labels} classes: {list(id2label.values())}")

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    ).to(device)

    # Create datasets
    train_dataset = CocoDetection(TRAIN_IMG_DIR, TRAIN_ANNO_FILE, processor)
    val_dataset = CocoDetection(VAL_IMG_DIR, VAL_ANNO_FILE, processor)

    # Collate function
    def collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        pixel_mask = torch.stack([item['pixel_mask'] for item in batch])
        labels = [item['labels'] for item in batch]
        return {'pixel_values': pixel_values, 'pixel_mask': pixel_mask, 'labels': labels}

    # Training arguments (compatible with transformers 4.55.1)
    training_args = TrainingArguments(
        output_dir="./detr_finetuned",
        num_train_epochs=50,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",  # Valid for 4.55.1
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False,
        logging_steps=10,
        save_total_limit=2
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=processor
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save model
    trainer.save_model("./detr_finetuned_model")
    processor.save_pretrained("./detr_finetuned_model")
    print("Model saved to ./detr_finetuned_model")

if __name__ == "__main__":
    main()
