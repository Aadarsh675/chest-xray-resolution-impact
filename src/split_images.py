import pandas as pd
import os
import json
import shutil
from sklearn.model_selection import train_test_split

# ==== CONFIGURATION ====
CSV_PATH = '/content/drive/My Drive/nih_chest_xray_dataset/BBox_List_2017.csv'
IMAGE_DIR = '/content/drive/My Drive/nih_chest_xray_dataset/images'
TRAIN_IMAGE_DIR = 'data/images/train'
TEST_IMAGE_DIR = 'data/images/test'
SPLIT_SAVE_DIR = 'data/splits'
TRAIN_SPLIT_FILE = os.path.join(SPLIT_SAVE_DIR, 'train_images.json')
TEST_SPLIT_FILE = os.path.join(SPLIT_SAVE_DIR, 'test_images.json')


def load_csv(csv_path):
    """Load CSV and return DataFrame."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


def split_images(df, train_ratio=0.8, random_state=42):
    """Split images into train and test sets and copy to respective directories."""
    # Get unique image indices
    unique_images = df['Image Index'].unique()
    print(f"Found {len(unique_images)} unique images")
    
    # Split images into train and test
    train_images, test_images = train_test_split(
        unique_images, 
        train_size=train_ratio, 
        random_state=random_state
    )
    
    # Create directories
    os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
    os.makedirs(TEST_IMAGE_DIR, exist_ok=True)
    os.makedirs(SPLIT_SAVE_DIR, exist_ok=True)
    
    # Copy images
    train_copied = 0
    test_copied = 0
    
    for img_name in train_images:
        src_path = os.path.join(IMAGE_DIR, img_name)
        dst_path = os.path.join(TRAIN_IMAGE_DIR, img_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            train_copied += 1
        else:
            print(f"Warning: Image {src_path} not found, skipping...")
    
    for img_name in test_images:
        src_path = os.path.join(IMAGE_DIR, img_name)
        dst_path = os.path.join(TEST_IMAGE_DIR, img_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            test_copied += 1
        else:
            print(f"Warning: Image {src_path} not found, skipping...")
    
    print(f"Copied {train_copied} images to {TRAIN_IMAGE_DIR}")
    print(f"Copied {test_copied} images to {TEST_IMAGE_DIR}")
    
    # Save split lists
    with open(TRAIN_SPLIT_FILE, 'w') as f:
        json.dump(list(train_images), f, indent=4)
    with open(TEST_SPLIT_FILE, 'w') as f:
        json.dump(list(test_images), f, indent=4)
    print(f"Saved train image list to {TRAIN_SPLIT_FILE}")
    print(f"Saved test image list to {TEST_SPLIT_FILE}")
    
    return train_images, test_images


def main():
    df = load_csv(CSV_PATH)
    train_images, test_images = split_images(df)


if __name__ == "__main__":
    main()
