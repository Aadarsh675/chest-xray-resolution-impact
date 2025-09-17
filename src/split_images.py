import pandas as pd
import os
import json
import shutil

# ==== CONFIGURATION ====
CSV_PATH = '/content/drive/My Drive/nih_chest_xray_dataset/BBox_List_2017.csv'  # kept for logging context
IMAGE_DIR = '/content/drive/My Drive/nih_chest_xray_dataset/images'

TRAIN_IMAGE_DIR = 'data/images/train'
TEST_IMAGE_DIR  = 'data/images/test'

SPLIT_SAVE_DIR = 'data/splits'
SHUFFLED_KEYS_FILE = os.path.join(SPLIT_SAVE_DIR, 'shuffled_image_keys.json')
TRAIN_SPLIT_FILE   = os.path.join(SPLIT_SAVE_DIR, 'train_images.json')
TEST_SPLIT_FILE    = os.path.join(SPLIT_SAVE_DIR, 'test_images.json')

TRAIN_RATIO = 0.8  # only used if train/test lists are missing


def load_csv(csv_path):
    """Load CSV and return DataFrame (optional, for info)."""
    if not os.path.exists(csv_path):
        print(f"[Info] CSV not found at {csv_path}. Skipping CSV-based stats.")
        return None
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


def load_split_lists():
    """
    Load train/test file lists if they exist. If not, reconstruct from shuffled keys + ratio.
    Returns (train_files, test_files).
    """
    if os.path.exists(TRAIN_SPLIT_FILE) and os.path.exists(TEST_SPLIT_FILE):
        with open(TRAIN_SPLIT_FILE, 'r') as f:
            train_files = json.load(f)
        with open(TEST_SPLIT_FILE, 'r') as f:
            test_files = json.load(f)
        print(f"Loaded explicit split lists: {len(train_files)} train, {len(test_files)} test.")
        return train_files, test_files

    # Fallback: use shuffled keys order + TRAIN_RATIO
    if os.path.exists(SHUFFLED_KEYS_FILE):
        with open(SHUFFLED_KEYS_FILE, 'r') as f:
            shuffled_keys = json.load(f)
        n_total = len(shuffled_keys)
        n_train = int(TRAIN_RATIO * n_total)
        train_files = shuffled_keys[:n_train]
        test_files = shuffled_keys[n_train:]
        print(f"[Fallback] Built split from shuffled keys: {len(train_files)} train, {len(test_files)} test.")
        return train_files, test_files

    raise FileNotFoundError(
        "No split lists found. Expected either "
        f"'{TRAIN_SPLIT_FILE}' & '{TEST_SPLIT_FILE}' or '{SHUFFLED_KEYS_FILE}'. "
        "Please run coco_converter.py first."
    )


def copy_images(file_list, src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    for img_name in file_list:
        src_path = os.path.join(src_dir, img_name)
        dst_path = os.path.join(dst_dir, img_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            copied += 1
        else:
            print(f"Warning: Image {src_path} not found, skipping...")
    return copied


def main():
    _ = load_csv(CSV_PATH)  # optional info

    # Load split based on saved key order (preferred) or explicit saved lists
    train_files, test_files = load_split_lists()

    # Copy images into train/test folders
    n_train = copy_images(train_files, IMAGE_DIR, TRAIN_IMAGE_DIR)
    n_test  = copy_images(test_files,  IMAGE_DIR, TEST_IMAGE_DIR)

    print(f"Copied {n_train} images to {TRAIN_IMAGE_DIR}")
    print(f"Copied {n_test} images to {TEST_IMAGE_DIR}")

    # Make sure we persist the final used lists (even if reconstructed from shuffled keys)
    os.makedirs(SPLIT_SAVE_DIR, exist_ok=True)
    with open(TRAIN_SPLIT_FILE, 'w') as f:
        json.dump(list(train_files), f, indent=4)
    with open(TEST_SPLIT_FILE, 'w') as f:
        json.dump(list(test_files), f, indent=4)
    print(f"Saved train image list to {TRAIN_SPLIT_FILE}")
    print(f"Saved test  image list to {TEST_SPLIT_FILE}")


if __name__ == "__main__":
    main()
