import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from PIL import Image

# ==== CONFIGURATION ====
CSV_PATH = '/content/drive/MyDrive/nih_chest_xray_dataset/BBox_List_2017.csv'
IMAGE_DIR = '/content/drive/MyDrive/nih_chest_xray_dataset/images'
SAVE_DIR = 'data/annotations'
SAVE_FILE = os.path.join(SAVE_DIR, 'annotations_coco.json')
TRAIN_SAVE_FILE = os.path.join(SAVE_DIR, 'train_annotations_coco.json')
TEST_SAVE_FILE = os.path.join(SAVE_DIR, 'test_annotations_coco.json')

def load_csv(csv_path):
    """Load CSV and return DataFrame."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df

def split_dataset(df, train_ratio=0.8, random_state=42):
    """Split DataFrame into train and test sets with stratification by disease label."""
    # Get unique images with their corresponding labels
    unique_images_df = df.groupby('Image Index')['Finding Label'].first().reset_index()
    print(f"Found {len(unique_images_df)} unique images")
   
    # Print number of images per disease label in the original dataset
    original_counts = unique_images_df['Finding Label'].value_counts()
    total_images = len(unique_images_df)
    print("\nOriginal dataset label distribution:")
    for label, count in original_counts.items():
        percentage = (count / total_images) * 100
        print(f"{label}: {count} images ({percentage:.2f}%)")
   
    # Split images into train and test, stratified by Finding Label
    train_images, test_images = train_test_split(
        unique_images_df['Image Index'],
        train_size=train_ratio,
        random_state=random_state,
        stratify=unique_images_df['Finding Label']  # Stratify by disease label
    )
   
    # Create train and test DataFrames
    train_df = df[df['Image Index'].isin(train_images)]
    test_df = df[df['Image Index'].isin(test_images)]
   
    print(f"\nTrain set: {len(train_df)} annotations for {len(train_images)} images")
    print(f"Test set: {len(test_df)} annotations for {len(test_images)} images")
   
    # Print number of images and percentages per disease label in train and test sets
    train_counts = train_df.groupby('Image Index')['Finding Label'].first().value_counts()
    test_counts = test_df.groupby('Image Index')['Finding Label'].first().value_counts()
    total_train_images = len(train_images)
    total_test_images = len(test_images)
   
    print("\nTrain set label distribution:")
    for label, count in train_counts.items():
        percentage = (count / total_train_images) * 100
        print(f"{label}: {count} images ({percentage:.2f}%)")
   
    print("\nTest set label distribution:")
    for label, count in test_counts.items():
        percentage = (count / total_test_images) * 100
        print(f"{label}: {count} images ({percentage:.2f}%)")
   
    return train_df, test_df

def parse_bbox(bbox_str):
    """Parse bounding box coordinates from the 'Bbox [x,y,w,h]' column."""
    # Remove brackets and split by comma
    bbox = bbox_str.strip('[]').split(',')
    return [float(coord) for coord in bbox]

def build_coco_format(df, image_dir):
    """Convert DataFrame to COCO-format dictionary, including image dimensions."""
    images = []
    annotations = []
    categories = []
    category_name_to_id = {}
    annotation_id = 1
    image_id_map = {}
    image_counter = 1

    for _, row in df.iterrows():
        file_name = row["Image Index"]
        # Assign a unique image_id and load dimensions
        if file_name not in image_id_map:
            img_path = os.path.join(image_dir, file_name)
            try:
                image = Image.open(img_path).convert("RGB")
                width, height = image.size
            except Exception as e:
                print(f"Error loading image {img_path}: {e}. Skipping...")
                continue

            image_id_map[file_name] = image_counter
            images.append({
                "id": image_counter,
                "file_name": file_name,
                "width": width,
                "height": height
            })
            image_counter += 1

        image_id = image_id_map[file_name]

        # Assign category id
        category_name = row["Finding Label"]
        if category_name not in category_name_to_id:
            category_id = len(category_name_to_id) + 1
            category_name_to_id[category_name] = category_id
            categories.append({
                "id": category_id,
                "name": category_name
            })
        else:
            category_id = category_name_to_id[category_name]

        # Parse bounding box coordinates
        x, y, w, h = parse_bbox(row['Bbox [x,y,w,h]'])

        # Create annotation entry
        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0
        })
        annotation_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    return coco_dict

def save_json(data, save_path):
    """Save dictionary as JSON."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"✅ COCO JSON saved to {save_path}")

def main():
    df = load_csv(CSV_PATH)
   
    # Split dataset into train and test
    train_df, test_df = split_dataset(df)
   
    # Convert to COCO format
    coco_data = build_coco_format(df, IMAGE_DIR)  # Original full dataset
    train_coco_data = build_coco_format(train_df, IMAGE_DIR)
    test_coco_data = build_coco_format(test_df, IMAGE_DIR)
   
    # Save JSON files
    save_json(coco_data, SAVE_FILE)  # Original full dataset
    save_json(train_coco_data, TRAIN_SAVE_FILE)
    save_json(test_coco_data, TEST_SAVE_FILE)

if __name__ == "__main__":
    main()
