import pandas as pd
import os
import json
import random
from sklearn.model_selection import train_test_split  # (kept import, not used now)

# ==== CONFIGURATION ====
CSV_PATH = '/content/drive/My Drive/nih_chest_xray_dataset/BBox_List_2017.csv'

SAVE_DIR = 'data/annotations'
SAVE_FILE = os.path.join(SAVE_DIR, 'annotations_coco.json')
TRAIN_SAVE_FILE = os.path.join(SAVE_DIR, 'train_annotations_coco.json')
TEST_SAVE_FILE = os.path.join(SAVE_DIR, 'test_annotations_coco.json')

SPLIT_SAVE_DIR = 'data/splits'
SHUFFLED_KEYS_FILE = os.path.join(SPLIT_SAVE_DIR, 'shuffled_image_keys.json')
TRAIN_SPLIT_FILE = os.path.join(SPLIT_SAVE_DIR, 'train_images.json')
TEST_SPLIT_FILE = os.path.join(SPLIT_SAVE_DIR, 'test_images.json')

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def load_csv(csv_path):
    """Load CSV and return DataFrame."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print("Columns:", df.columns.tolist())
    return df


def build_coco_full(df):
    """
    Convert the entire DataFrame to a single COCO-format dictionary (no splitting yet).
    """
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
                "width": None,   # Optional: fill actual sizes later if needed
                "height": None
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

        # BBox columns in your CSV
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
        "info": {},       # kept minimal but valid
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    return coco_dict


def save_json(data, save_path):
    """Save dictionary or list as JSON."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"✅ Saved to {save_path}")


def filter_coco_by_images(coco_full, keep_file_names_set):
    """
    Create a new COCO dict containing only images whose file_name is in keep_file_names_set,
    and only annotations that reference those images.
    """
    # Map from file_name -> image object (and id)
    file_to_img = {img["file_name"]: img for img in coco_full["images"]}
    keep_images = [file_to_img[f] for f in file_to_img if f in keep_file_names_set]
    keep_img_ids = set(img["id"] for img in keep_images)

    # Keep only annotations for those image ids
    keep_annotations = [ann for ann in coco_full["annotations"] if ann["image_id"] in keep_img_ids]

    # Rebuild images list (optional: preserve shuffled order later when saving)
    coco_new = {
        "info": coco_full.get("info", {}),
        "licenses": coco_full.get("licenses", []),
        "images": keep_images,
        "annotations": keep_annotations,
        "categories": coco_full["categories"]
    }
    return coco_new


def main():
    random.seed(RANDOM_SEED)

    df = load_csv(CSV_PATH)

    # 1) Build full (unsplit) COCO
    coco_full = build_coco_full(df)
    print(f"Full COCO: {len(coco_full['images'])} images, {len(coco_full['annotations'])} annotations.")

    # 2) Randomly shuffle images (by file_name) — prevents order by label
    all_file_names = [img["file_name"] for img in coco_full["images"]]
    random.shuffle(all_file_names)  # in-place shuffle using fixed seed (set above)
    print("Shuffled image order created.")

    # 3) Save the shuffled key order (file_name list)
    os.makedirs(SPLIT_SAVE_DIR, exist_ok=True)
    save_json(all_file_names, SHUFFLED_KEYS_FILE)

    # 4) Split into train/test by this shuffled order
    n_total = len(all_file_names)
    n_train = int(TRAIN_RATIO * n_total)
    train_files = all_file_names[:n_train]
    test_files = all_file_names[n_train:]

    # Save split lists (explicit)
    save_json(train_files, TRAIN_SPLIT_FILE)
    save_json(test_files, TEST_SPLIT_FILE)

    # 5) Build train/test COCO dicts from full by filtering with the split file names
    train_coco = filter_coco_by_images(coco_full, set(train_files))
    test_coco  = filter_coco_by_images(coco_full, set(test_files))

    # (Optional) To keep the "images" array in the *shuffled* order,
    # reorder images to follow the saved sequence:
    train_coco["images"] = [img for fn in train_files for img in train_coco["images"] if img["file_name"] == fn]
    test_coco["images"]  = [img for fn in test_files  for img in test_coco["images"]  if img["file_name"] == fn]

    # 6) Save JSONs
    save_json(coco_full, SAVE_FILE)
    save_json(train_coco, TRAIN_SAVE_FILE)
    save_json(test_coco, TEST_SAVE_FILE)

    print(f"Train split: {len(train_coco['images'])} images, {len(train_coco['annotations'])} anns.")
    print(f"Test split:  {len(test_coco['images'])} images, {len(test_coco['annotations'])} anns.")


if __name__ == "__main__":
    main()
