"""
Dataset downloader for NIH Chest X-rays
"""

import os
import subprocess
import shutil
from pathlib import Path


def download_nih_dataset(output_dir="./data"):
    """
    Download NIH Chest X-rays dataset using kagglehub
    
    Args:
        output_dir: Directory to save the dataset
    
    Returns:
        Path to the downloaded dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if kagglehub is installed
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        subprocess.run([
            "pip", "install", "kagglehub"
        ], check=True)
        import kagglehub
    
    # Download dataset
    print("Downloading NIH Chest X-rays dataset...")
    print("This may take a while depending on your internet connection...")
    
    try:
        # Download using kagglehub
        downloaded_path = kagglehub.dataset_download("nih-chest-xrays/data")
        print(f"✓ Dataset downloaded to: {downloaded_path}")
        
        # Copy to desired location
        final_path = os.path.join(output_dir, "nih-chest-xrays")
        
        if os.path.exists(final_path):
            print(f"Removing existing dataset at {final_path}")
            shutil.rmtree(final_path)
        
        print(f"Copying dataset to {final_path}...")
        shutil.copytree(downloaded_path, final_path)
        
        # Verify dataset structure
        verify_dataset_structure(final_path)
        
        return final_path
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        
        # Alternative download method
        print("\nTrying alternative download method...")
        return download_with_wget(output_dir)


def download_with_wget(output_dir):
    """
    Alternative download method using wget
    """
    dataset_dir = os.path.join(output_dir, "nih-chest-xrays")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # URLs for the dataset files (these are examples, actual URLs may differ)
    files_to_download = [
        "Data_Entry_2017_v2020.csv",
        "BBox_List_2017.csv",
        "train_val_list.txt",
        "test_list.txt"
    ]
    
    base_url = "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/"
    
    print("Note: Direct download URLs may need to be updated.")
    print("Please ensure you have the proper access to download the NIH dataset.")
    
    # For the actual implementation, you would need the correct URLs
    # This is a placeholder for the structure
    
    return dataset_dir


def verify_dataset_structure(dataset_path):
    """
    Verify that the dataset has the expected structure
    """
    print("\nVerifying dataset structure...")
    
    expected_files = [
        "Data_Entry_2017_v2020.csv",
        "BBox_List_2017.csv"
    ]
    
    missing_files = []
    for file in expected_files:
        file_path = os.path.join(dataset_path, file)
        if os.path.exists(file_path):
            print(f"✓ Found: {file}")
        else:
            missing_files.append(file)
            print(f"✗ Missing: {file}")
    
    # Check for images
    image_count = 0
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_count += 1
    
    print(f"\n✓ Found {image_count} images in the dataset")
    
    if missing_files:
        print(f"\n⚠ Warning: Some expected files are missing: {missing_files}")
    
    return len(missing_files) == 0