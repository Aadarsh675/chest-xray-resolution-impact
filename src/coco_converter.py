import pandas as pd
import os
import json

# ==== CONFIGURATION ====
CSV_PATH = '/content/drive/My Drive/nih_chest_xray_dataset/BBox_List_2017.csv'
SAVE_DIR = 'data/annotations'
SAVE_FILE = os.path.join(SAVE_DIR, 'annotations_coco.json')


def load_csv(csv_path):
    """Load CSV and return DataFrame."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


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
                "width": None,  # Optional: set actual width if available
                "height": None  # Optional: set actual height if available
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

        # Create annotation entry
        bbox = [
            row["Bbox [x"],  # x
            row["y"],        # y
            row["w"],        # width
            row["h]"]        # height
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
    coco_data = build_coco_format(df)
    save_json(coco_data, SAVE_FILE)


if __name__ == "__main__":
    main()
