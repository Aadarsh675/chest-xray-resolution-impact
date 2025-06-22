import os
import kagglehub
import shutil
import pandas as pd

# Define custom target path inside your GitHub repo
TARGET_PATH = "medicalImageSuperResolution/dataset/standard/train_images"
EXPECTED_FILE = os.path.join(TARGET_PATH, "00000001_000.png")  # any known file from dataset

# Check if already downloaded
def dataset_already_downloaded(path):
    return os.path.exists(path) and len(os.listdir(path)) > 0

# Download if needed
if dataset_already_downloaded(TARGET_PATH):
    print(f"✅ Dataset already exists at {TARGET_PATH}")
else:
    print(f"⬇️ Downloading NIH dataset to {TARGET_PATH}...")
    path = kagglehub.dataset_download("nih-chest-xrays/data", path=TARGET_PATH)
    print("✅ Download complete.")

# Paths
ALL_IMAGES = "dataset/standard/train_images"  # or wherever you downloaded all NIH images
CLASS_DIR = "dataset/classification/images"
BBOX_DIR = "dataset/bbox/images"

os.makedirs(CLASS_DIR, exist_ok=True)
os.makedirs(BBOX_DIR, exist_ok=True)

# Load CSVs
entry_df = pd.read_csv(os.path.join(ALL_IMAGES, "../Data_Entry_2017.csv"))
bbox_df = pd.read_csv(os.path.join(ALL_IMAGES, "../BBox_List_2017.csv"))

# Get filenames with bounding boxes
bbox_filenames = set(bbox_df['Image Index'])

# Organize files
classification_rows = []
bbox_rows = []

for idx, row in entry_df.iterrows():
    filename = row["Image Index"]
    label = row["Finding Labels"]

    src = os.path.join(ALL_IMAGES, filename)

    if filename in bbox_filenames:
        dst = os.path.join(BBOX_DIR, filename)
        bbox_rows.append(row)
    else:
        dst = os.path.join(CLASS_DIR, filename)
        classification_rows.append(row)

    if os.path.exists(src):
        shutil.copy(src, dst)

# Save filtered CSVs
pd.DataFrame(classification_rows).to_csv("dataset/classification/labels.csv", index=False)
pd.DataFrame(bbox_rows).to_csv("dataset/bbox/labels.csv", index=False)

print("✅ Dataset organized into 'classification/' and 'bbox/' folders.")
