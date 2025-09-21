import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split

# ==== CONFIGURATION ====
CSV_PATH = '/content/drive/My Drive/nih_chest_xray_dataset/BBox_List_2017.csv'
SAVE_DIR = 'data/annotations'
SAVE_FILE = os.path.join(SAVE_DIR, 'annotations_coco.json')
TRAIN_SAVE_FILE = os.path.join(SAVE_DIR, 'train_annotations_coco.json')
TEST_SAVE_FILE = os.path.join(SAVE_DIR, 'test_annotations_coco.json')


def load_csv(csv_path):
    """Load CSV and return DataFrame."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print("Columns:", df.columns.tolist())
    return df


def split_dataset(df, train_ratio=0.8, random_state=42):
    """Split DataFrame into train and test sets with stratification by disease label."""
    # Get unique images with their corresponding labels
    unique_images_df = df.groupby('Image Index')['Finding Label'].first().reset_index()
    print(f"Found {len(unique_images_df)} unique images")

    # Print number of images per disease label
    original_counts = unique_images_df['Finding Label'].value_counts()
    total_images = len(unique_images_df)
    print("\nOriginal dataset label distribution:")
    for label, count in original_counts.items():
        print(f"{label}: {count} images ({(count/total_images)*100:.2f}%)")

    # Split images
    train_images, test_images = train_test_split(
        unique_images_df['Image Index'],
        train_size=train_ratio,
        random_state=random_state,
        stratify=unique_images_df['Finding Label']
    )

    train_df = df[df['Image Index'].isin(train_images)]
    test_df = df[df['Image Index'].isin(test_images)]

    print(f"\nTrain set: {len(train_df)} annotations for {len(train_images)} images")
    print(f"Test set: {len(test_df)} annotations for {len(test_images)} images")

    # Print label distribution in splits
    train_counts = train_df.groupby('Image Index')['Finding Label'].first().value_counts()
    test_counts = test_df.groupby('Image Index')['Finding Label'].first().value_counts()

    print("\nTrain set label distribution:")
    for label, count in train_counts.items():
        print(f"{label}: {count} images ({(count/len(train_images))*100:.2f}%)")

    print("\nTest set label distribution:")
    for label, count in test_counts.items():
        print(f"{label}: {count} images ({(count/len(test_images))*100:.2f}%)")

    return train_df, test_df


def build_coco_format(df):
    """Convert DataFrame to COCO-format dictionary."""
    images = []
    annotations = []
    categories = []
    category_name_to_id = {}
    annotation_id = 1
    image_id_map = {}
    image_counter = 1

    for _, row in df.iterrows():
        file_name = row["Image Index"]

        # Assign a unique image_id
        if file_name not in image_id_map:
            image_id_map[file_name] = image_counter
            images.append({
                "id": image_counter,
                "file_name": file_name,
                "width": None,  # Optional: fill with actual image size
                "height": None  # Optional: fill with actual image size
            })
            image_counter += 1
        image_id = image_id_map[file_name]

        # Assign category id
        category_name = row["Finding Label"]
        if category_name not in category_name_to_id:
            category_id = len(category_name_to_id) + 1
            category_name_to_id[category_name] = category_id
            categories.append({"id": category_id, "name": category_name})
        else:
            category_id = category_name_to_id[category_name]

        # Get bbox from separate columns
        bbox = [
            float(row['Bbox [x']),  # x
            float(row['y']),        # y
            float(row['w']),        # width
            float(row['h]'])        # height
        ]

        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        })
        annotation_id += 1

    coco_dict = {
        "info": {},       # Empty but required
        "licenses": [],   # Empty but required
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

    # Build COCO datasets
    coco_data = build_coco_format(df)
    train_coco_data = build_coco_format(train_df)
    test_coco_data = build_coco_format(test_df)

    # Save JSONs
    save_json(coco_data, SAVE_FILE)
    save_json(train_coco_data, TRAIN_SAVE_FILE)
    save_json(test_coco_data, TEST_SAVE_FILE)


if __name__ == "__main__":
    main()
