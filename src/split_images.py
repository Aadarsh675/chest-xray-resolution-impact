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
    """Split images into train and test sets with stratification by disease label and copy to respective directories."""
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
    
    # Create train and test DataFrames for label distribution
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
    
    # Create directories
    os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
    os.makedirs(TEST_IMAGE_DIR, exist_ok=True)
    os.makedirs(SPLIT_SAVE_DIR, exist_ok=True)
    
    # Copy images
    train_copied = 0
    test_copied = 0

    total_train = len(train_images)
    total_test = len(test_images)
    print(f"\nCopying train images ({total_train} total)...")
    for img_name in train_images:
        src_path = os.path.join(IMAGE_DIR, img_name)
        dst_path = os.path.join(TRAIN_IMAGE_DIR, img_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            train_copied += 1
            # progress print
            if train_copied % 100 == 0 or train_copied == total_train:
                print(f"  Train progress: {train_copied}/{total_train} ({train_copied/total_train*100:.1f}%)")
        else:
            print(f"Warning: Image {src_path} not found, skipping...")
    
    print(f"\nCopying test images ({total_test} total)...")
    for img_name in test_images:
        src_path = os.path.join(IMAGE_DIR, img_name)
        dst_path = os.path.join(TEST_IMAGE_DIR, img_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            test_copied += 1
            # progress print
            if test_copied % 100 == 0 or test_copied == total_test:
                print(f"  Test progress: {test_copied}/{total_test} ({test_copied/total_test*100:.1f}%)")
        else:
            print(f"Warning: Image {src_path} not found, skipping...")
    
    print(f"\nCopied {train_copied} images to {TRAIN_IMAGE_DIR}")
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
