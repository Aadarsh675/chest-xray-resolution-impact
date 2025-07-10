"""
Google Drive handler for dataset management
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime


class GDriveHandler:
    def __init__(self, gdrive_base_path='/content/drive/MyDrive/nih-chest-xrays'):
        """
        Initialize Google Drive handler
        
        Args:
            gdrive_base_path: Base path in Google Drive for dataset storage
        """
        self.gdrive_base_path = gdrive_base_path
        self.drive_mount_path = '/content/drive'
        
    def is_mounted(self):
        """Check if Google Drive is mounted"""
        # Check for Colab environment
        try:
            from google.colab import drive
            if not os.path.exists(self.drive_mount_path):
                print("Mounting Google Drive...")
                drive.mount(self.drive_mount_path)
            return True
        except ImportError:
            # Not in Colab
            return os.path.exists(self.drive_mount_path)
    
    def dataset_exists(self):
        """Check if dataset exists in Google Drive"""
        if not self.is_mounted():
            return False
        
        # Check for key dataset files
        required_files = [
            'Data_Entry_2017_v2020.csv',
            'BBox_List_2017.csv'
        ]
        
        for file in required_files:
            file_path = os.path.join(self.gdrive_base_path, file)
            if not os.path.exists(file_path):
                return False
        
        # Check if images exist
        images_dir = os.path.join(self.gdrive_base_path, 'images')
        if not os.path.exists(images_dir):
            # Check alternative locations
            for subdir in ['images_001', 'images_002', 'images']:
                alt_path = os.path.join(self.gdrive_base_path, subdir)
                if os.path.exists(alt_path):
                    return True
            return False
        
        return True
    
    def get_dataset_path(self):
        """Get the dataset path in Google Drive"""
        if self.dataset_exists():
            return self.gdrive_base_path
        return None
    
    def upload_dataset(self, local_dataset_path):
        """
        Upload dataset from local path to Google Drive
        
        Args:
            local_dataset_path: Local path containing the dataset
        """
        if not self.is_mounted():
            print("Google Drive not mounted, skipping upload")
            return False
        
        print(f"Uploading dataset to Google Drive: {self.gdrive_base_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(self.gdrive_base_path, exist_ok=True)
        
        # Create metadata file
        metadata = {
            'upload_date': datetime.now().isoformat(),
            'source_path': str(local_dataset_path),
            'dataset': 'NIH Chest X-rays'
        }
        
        metadata_path = os.path.join(self.gdrive_base_path, 'dataset_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Copy files with progress tracking
        total_files = sum([len(files) for _, _, files in os.walk(local_dataset_path)])
        copied_files = 0
        
        print(f"Copying {total_files} files...")
        
        for root, dirs, files in os.walk(local_dataset_path):
            # Calculate relative path
            rel_path = os.path.relpath(root, local_dataset_path)
            dest_dir = os.path.join(self.gdrive_base_path, rel_path)
            
            # Create directory
            os.makedirs(dest_dir, exist_ok=True)
            
            # Copy files
            for file in files:
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)
                
                # Skip if file already exists and has same size
                if os.path.exists(dest_file):
                    if os.path.getsize(src_file) == os.path.getsize(dest_file):
                        copied_files += 1
                        continue
                
                try:
                    shutil.copy2(src_file, dest_file)
                    copied_files += 1
                    
                    if copied_files % 100 == 0:
                        print(f"Progress: {copied_files}/{total_files} files copied")
                        
                except Exception as e:
                    print(f"Error copying {file}: {e}")
        
        print(f"✓ Dataset uploaded successfully! ({copied_files} files)")
        return True
    
    def download_from_gdrive(self, local_path):
        """
        Download dataset from Google Drive to local path
        
        Args:
            local_path: Local directory to save the dataset
        """
        if not self.dataset_exists():
            print("Dataset not found in Google Drive")
            return False
        
        print(f"Downloading dataset from Google Drive to {local_path}")
        
        # Create local directory
        os.makedirs(local_path, exist_ok=True)
        
        # Copy dataset
        try:
            # Use rsync for efficient copying if available
            if shutil.which('rsync'):
                import subprocess
                subprocess.run([
                    'rsync', '-av', '--progress',
                    f'{self.gdrive_base_path}/',
                    f'{local_path}/'
                ], check=True)
            else:
                # Fallback to shutil
                shutil.copytree(
                    self.gdrive_base_path,
                    local_path,
                    dirs_exist_ok=True
                )
            
            print("✓ Dataset downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    
    def upload_results(self, results_dir):
        """
        Upload training results to Google Drive
        
        Args:
            results_dir: Directory containing results to upload
        """
        if not self.is_mounted():
            print("Google Drive not mounted, skipping results upload")
            return False
        
        # Create results directory in Google Drive
        gdrive_results_dir = os.path.join(
            self.gdrive_base_path, 
            f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        
        os.makedirs(gdrive_results_dir, exist_ok=True)
        
        print(f"Uploading results to: {gdrive_results_dir}")
        
        # Copy results
        try:
            shutil.copytree(
                results_dir,
                gdrive_results_dir,
                dirs_exist_ok=True
            )
            print("✓ Results uploaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error uploading results: {e}")
            return False
    
    def list_datasets(self):
        """List all datasets in Google Drive"""
        if not self.is_mounted():
            return []
        
        datasets = []
        
        if os.path.exists(self.gdrive_base_path):
            # Look for dataset metadata files
            for item in os.listdir(self.gdrive_base_path):
                item_path = os.path.join(self.gdrive_base_path, item)
                metadata_path = os.path.join(item_path, 'dataset_metadata.json')
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        datasets.append({
                            'path': item_path,
                            'metadata': metadata
                        })
        
        return datasets