import os
import kagglehub

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

