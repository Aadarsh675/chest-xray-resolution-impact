@@ -13,27 +13,57 @@
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

    # Split images into train and test
    # Split images into train and test, stratified by Finding Label
    train_images, test_images = train_test_split(
        unique_images, 
        train_size=train_ratio, 
        random_state=random_state
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
@@ -74,11 +104,9 @@ def split_images(df, train_ratio=0.8, random_state=42):

    return train_images, test_images


def main():
    df = load_csv(CSV_PATH)
    train_images, test_images = split_images(df)


if __name__ == "__main__":
    main()
