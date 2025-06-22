import os
import json
import shutil
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np


class NIHDatasetOrganizer:
    """Organizes NIH Chest X-ray dataset into classification and bbox subsets with visualization."""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = base_path
        self.raw_path = os.path.join(base_path, "raw/nih-chest-xrays")
        self.raw_images_path = os.path.join(self.raw_path, "images")
        self.class_dir = os.path.join(base_path, "processed/classification")
        self.bbox_dir = os.path.join(base_path, "processed/bbox")
        self.plots_dir = os.path.join(base_path, "plots")
        
        # Expected files for checking if dataset exists
        self.expected_file = os.path.join(self.raw_images_path, "00000001_000.png")
        
        # Disease categories for COCO format
        self.disease_categories = [
            "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
            "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
            "Pleural_thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"
        ]
        
        # Create all necessary directories
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def dataset_already_downloaded(self) -> bool:
        """Check if dataset is already downloaded."""
        # Check for CSV files which are essential
        csv_files = ["Data_Entry_2017.csv", "BBox_List_2017.csv"]
        
        # Look in various possible locations
        possible_paths = [
            self.raw_path,
            os.path.join(self.raw_path, "data"),
            os.path.join(self.raw_path, "archive")
        ]
        
        csv_found = False
        images_found = False
        
        # Check for CSV files
        for path in possible_paths:
            if all(os.path.exists(os.path.join(path, csv)) for csv in csv_files):
                csv_found = True
                break
        
        # Check for image files
        image_patterns = ["images", "images_001", "png_images", "data"]
        for base in possible_paths:
            for pattern in image_patterns:
                img_path = os.path.join(base, pattern)
                if os.path.exists(img_path) and os.path.isdir(img_path):
                    # Check if it contains PNG files
                    files = os.listdir(img_path)
                    if any(f.endswith('.png') for f in files[:10]):
                        images_found = True
                        break
            if images_found:
                break
        
        return csv_found and images_found
    
    def download_dataset(self):
        """Download NIH dataset if not already present."""
        if self.dataset_already_downloaded():
            print(f"✅ Dataset already exists at {self.raw_images_path}")
            return
        
        print(f"⬇️ Downloading NIH dataset to {self.raw_path}...")
        
        # Try different dataset identifiers
        dataset_identifiers = [
            "nih-chest-xrays/data",
            "nih-chest-xrays/nih-chest-xrays",
            "nihchestxrays/data",
            "nih/chest-xrays"
        ]
        
        download_success = False
        last_error = None
        
        for identifier in dataset_identifiers:
            try:
                print(f"   Trying identifier: {identifier}")
                path = kagglehub.dataset_download(identifier, path=self.raw_path)
                print("✅ Download complete.")
                download_success = True
                break
            except Exception as e:
                last_error = e
                print(f"   ❌ Failed with: {identifier}")
                continue
        
        if not download_success:
            print(f"\n❌ Could not download dataset automatically.")
            print(f"Last error: {last_error}")
            print("\n📋 Manual download instructions:")
            print("1. Visit: https://www.kaggle.com/datasets/nih-chest-xrays/data")
            print("2. Click 'Download' button (requires Kaggle account)")
            print("3. Extract the downloaded archive to: " + self.raw_path)
            print("4. Ensure the images are in: " + self.raw_images_path)
            print("\nAlternatively, try running:")
            print("   kaggle datasets download -d nih-chest-xrays/data")
            print("   unzip -q data.zip -d " + self.raw_path)
            
            raise Exception("Failed to download dataset. See instructions above.")
    
    def load_metadata(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load the CSV files containing labels and bbox information."""
        # Try different possible locations for the CSV files
        possible_paths = [
            self.raw_path,
            os.path.join(self.raw_path, "data"),
            os.path.dirname(self.raw_images_path),
            os.path.join(self.raw_path, "archive"),  # Sometimes in archive folder
            self.raw_images_path  # Sometimes CSVs are with images
        ]
        
        # Also look for the image folders
        image_folders = ["images", "images_001", "png_images", "data"]
        
        entry_df = None
        bbox_df = None
        
        # First, try to find where images actually are
        for base in possible_paths:
            for img_folder in image_folders:
                test_path = os.path.join(base, img_folder)
                if os.path.exists(test_path) and os.path.isdir(test_path):
                    # Check if it contains images
                    files = os.listdir(test_path)
                    if any(f.endswith('.png') for f in files[:10]):  # Check first 10 files
                        self.raw_images_path = test_path
                        print(f"✅ Found images at: {test_path}")
                        break
        
        # Now look for CSV files
        for path in possible_paths:
            entry_path = os.path.join(path, "Data_Entry_2017.csv")
            bbox_path = os.path.join(path, "BBox_List_2017.csv")
            
            if os.path.exists(entry_path) and entry_df is None:
                entry_df = pd.read_csv(entry_path)
                print(f"✅ Loaded Data_Entry_2017.csv from {path}")
                
            if os.path.exists(bbox_path) and bbox_df is None:
                bbox_df = pd.read_csv(bbox_path)
                print(f"✅ Loaded BBox_List_2017.csv from {path}")
        
        if entry_df is None or bbox_df is None:
            print("\n❌ Could not find required CSV files")
            print("Looking for:")
            print("  - Data_Entry_2017.csv")
            print("  - BBox_List_2017.csv")
            print("\nSearched in:")
            for path in possible_paths:
                print(f"  - {path}")
            print("\nPlease ensure the dataset is properly extracted.")
            raise FileNotFoundError("Could not find required CSV files")
            
        return entry_df, bbox_df
    
    def parse_finding_labels(self, labels_str: str) -> List[str]:
        """Parse the finding labels string into a list of diseases."""
        if pd.isna(labels_str) or labels_str == "No Finding":
            return []
        return [label.strip() for label in labels_str.split("|")]
    
    def plot_disease_distribution(self, entry_df: pd.DataFrame):
        """Plot distribution of diseases across all images."""
        # Count disease occurrences
        all_diseases = []
        for labels in entry_df['Finding Labels']:
            all_diseases.extend(self.parse_finding_labels(labels))
        
        disease_counts = Counter(all_diseases)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        diseases = list(disease_counts.keys())
        counts = list(disease_counts.values())
        
        plt.barh(diseases, counts, color='skyblue', edgecolor='navy')
        plt.xlabel('Number of Images', fontsize=12)
        plt.ylabel('Disease Type', fontsize=12)
        plt.title('Disease Distribution in NIH Chest X-ray Dataset', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add count labels
        for i, (disease, count) in enumerate(zip(diseases, counts)):
            plt.text(count + 100, i, f'{count:,}', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'disease_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return disease_counts
    
    def plot_bbox_vs_classification_split(self, n_class: int, n_bbox: int):
        """Plot pie chart showing split between classification and bbox images."""
        plt.figure(figsize=(8, 8))
        
        labels = ['Classification Only', 'With Bounding Boxes']
        sizes = [n_class, n_bbox]
        colors = ['#ff9999', '#66b3ff']
        explode = (0.05, 0.05)
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90, textprops={'fontsize': 12})
        
        plt.title('Dataset Split: Classification vs Bounding Box Annotations', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Add counts in legend
        plt.legend([f'{labels[0]}: {sizes[0]:,}', f'{labels[1]}: {sizes[1]:,}'], 
                  loc='best', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'dataset_split.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_bbox_disease_distribution(self, coco_data: Dict):
        """Plot distribution of diseases in bbox annotations."""
        # Count annotations per disease
        disease_counts = {}
        for ann in coco_data['annotations']:
            cat_id = ann['category_id']
            cat_name = next(c['name'] for c in coco_data['categories'] if c['id'] == cat_id)
            disease_counts[cat_name] = disease_counts.get(cat_name, 0) + 1
        
        # Create plot
        plt.figure(figsize=(10, 6))
        diseases = list(disease_counts.keys())
        counts = list(disease_counts.values())
        
        bars = plt.bar(diseases, counts, color='coral', edgecolor='darkred')
        plt.xlabel('Disease Type', fontsize=12)
        plt.ylabel('Number of Bounding Boxes', fontsize=12)
        plt.title('Bounding Box Annotations by Disease Type', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'bbox_disease_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return disease_counts
    
    def plot_patient_demographics(self, entry_df: pd.DataFrame):
        """Plot patient age and gender distributions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Age distribution
        ages = entry_df['Patient Age'].dropna()
        ax1.hist(ages, bins=20, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        ax1.set_xlabel('Age (years)', fontsize=12)
        ax1.set_ylabel('Number of Images', fontsize=12)
        ax1.set_title('Patient Age Distribution', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add statistics
        ax1.axvline(ages.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ages.mean():.1f}')
        ax1.axvline(ages.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {ages.median():.1f}')
        ax1.legend()
        
        # Gender distribution
        gender_counts = entry_df['Patient Gender'].value_counts()
        ax2.pie(gender_counts.values, labels=['Male', 'Female'], autopct='%1.1f%%',
                colors=['#87CEEB', '#FFB6C1'], startangle=90, textprops={'fontsize': 12})
        ax2.set_title('Patient Gender Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'patient_demographics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_multi_label_statistics(self, entry_df: pd.DataFrame):
        """Plot statistics about multi-label classifications."""
        # Count number of diseases per image
        disease_counts_per_image = []
        for labels in entry_df['Finding Labels']:
            diseases = self.parse_finding_labels(labels)
            disease_counts_per_image.append(len(diseases))
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        unique_counts = list(range(0, max(disease_counts_per_image) + 1))
        count_freq = [disease_counts_per_image.count(i) for i in unique_counts]
        
        bars = plt.bar(unique_counts, count_freq, color='mediumpurple', edgecolor='darkblue')
        plt.xlabel('Number of Diseases per Image', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.title('Multi-Label Distribution: Diseases per Image', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        total_images = len(disease_counts_per_image)
        for bar, freq in zip(bars, count_freq):
            if freq > 0:
                percentage = (freq / total_images) * 100
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
                        f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'multi_label_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print statistics
        print(f"\n📊 Multi-label Statistics:")
        print(f"   Images with 0 diseases (No Finding): {disease_counts_per_image.count(0):,}")
        print(f"   Images with 1 disease: {disease_counts_per_image.count(1):,}")
        print(f"   Images with 2+ diseases: {sum(1 for c in disease_counts_per_image if c >= 2):,}")
        print(f"   Max diseases in single image: {max(disease_counts_per_image)}")
    
    def organize_classification_data(self, entry_df: pd.DataFrame, 
                                   bbox_filenames: set) -> pd.DataFrame:
        """
        Organize images without bounding boxes into classification folder.
        Returns DataFrame with classification data.
        """
        class_images_dir = os.path.join(self.class_dir, "images")
        os.makedirs(class_images_dir, exist_ok=True)
        
        classification_rows = []
        copied_count = 0
        
        for idx, row in entry_df.iterrows():
            filename = row["Image Index"]
            
            # Skip if this image has bounding boxes
            if filename in bbox_filenames:
                continue
                
            src = os.path.join(self.raw_images_path, filename)
            dst = os.path.join(class_images_dir, filename)
            
            if os.path.exists(src):
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied_count += 1
                classification_rows.append(row)
        
        # Save classification labels
        class_df = pd.DataFrame(classification_rows)
        class_df.to_csv(os.path.join(self.class_dir, "labels.csv"), index=False)
        
        print(f"✅ Organized {len(classification_rows)} classification images")
        print(f"   Copied {copied_count} new images to {class_images_dir}")
        
        return class_df
    
    def create_coco_format(self, bbox_df: pd.DataFrame, 
                          entry_df: pd.DataFrame) -> Dict:
        """
        Convert bounding box annotations to COCO format.
        """
        # Create category mapping
        categories = []
        category_to_id = {}
        
        for idx, disease in enumerate(self.disease_categories):
            categories.append({
                "id": idx + 1,
                "name": disease,
                "supercategory": "thorax_disease"
            })
            category_to_id[disease] = idx + 1
        
        # Process images and annotations
        images = []
        annotations = []
        image_to_id = {}
        annotation_id = 1
        
        # Get unique images with bboxes
        bbox_images = bbox_df['Image Index'].unique()
        
        for img_idx, image_name in enumerate(bbox_images):
            # Image info (NIH images are 1024x1024)
            image_id = img_idx + 1
            image_to_id[image_name] = image_id
            
            images.append({
                "id": image_id,
                "file_name": image_name,
                "width": 1024,
                "height": 1024
            })
            
            # Get all bboxes for this image
            image_bboxes = bbox_df[bbox_df['Image Index'] == image_name]
            
            for _, bbox_row in image_bboxes.iterrows():
                # Parse bbox coordinates
                x = float(bbox_row['Bbox [x'])
                y = float(bbox_row['y'])
                w = float(bbox_row['w'])
                h = float(bbox_row['h]'])
                
                # Get category
                disease = bbox_row['Finding Label']
                if disease in category_to_id:
                    category_id = category_to_id[disease]
                else:
                    # Handle disease names that might not match exactly
                    continue
                
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                annotation_id += 1
        
        coco_format = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
            "info": {
                "description": "NIH Chest X-ray Dataset with Bounding Box Annotations",
                "version": "1.0",
                "year": 2017,
                "contributor": "National Institutes of Health",
                "date_created": "2017"
            }
        }
        
        return coco_format
    
    def organize_bbox_data(self, bbox_df: pd.DataFrame, 
                          entry_df: pd.DataFrame) -> Dict:
        """
        Organize images with bounding boxes and create COCO format annotations.
        Returns COCO format dictionary.
        """
        bbox_images_dir = os.path.join(self.bbox_dir, "images")
        os.makedirs(bbox_images_dir, exist_ok=True)
        
        # Get unique images with bboxes
        bbox_filenames = set(bbox_df['Image Index'])
        copied_count = 0
        
        # Copy bbox images
        for filename in bbox_filenames:
            src = os.path.join(self.raw_images_path, filename)
            dst = os.path.join(bbox_images_dir, filename)
            
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied_count += 1
        
        # Create COCO format annotations
        coco_data = self.create_coco_format(bbox_df, entry_df)
        
        # Save COCO format JSON
        coco_path = os.path.join(self.bbox_dir, "annotations_coco.json")
        with open(coco_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        # Also save the original bbox CSV for reference
        bbox_df.to_csv(os.path.join(self.bbox_dir, "bbox_original.csv"), index=False)
        
        print(f"✅ Organized {len(bbox_filenames)} bbox images")
        print(f"   Copied {copied_count} new images to {bbox_images_dir}")
        print(f"   Created COCO format with {len(coco_data['annotations'])} annotations")
        print(f"   Saved to {coco_path}")
        
        return coco_data
    
    def generate_summary_report(self, entry_df: pd.DataFrame, bbox_df: pd.DataFrame,
                               class_df: pd.DataFrame, coco_data: Dict):
        """Generate a summary report of the dataset organization."""
        report_path = os.path.join(self.base_path, "dataset_summary.txt")
        
        with open(report_path, 'w') as f:
            f.write("NIH CHEST X-RAY DATASET SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total images: {len(entry_df):,}\n")
            f.write(f"Unique patients: {entry_df['Patient ID'].nunique():,}\n")
            f.write(f"Images with bounding boxes: {len(bbox_df['Image Index'].unique()):,}\n")
            f.write(f"Images for classification only: {len(class_df):,}\n")
            f.write(f"Total bounding box annotations: {len(coco_data['annotations']):,}\n\n")
            
            f.write("FOLDER STRUCTURE\n")
            f.write("-" * 30 + "\n")
            f.write(f"Raw data: {self.raw_path}\n")
            f.write(f"Classification data: {self.class_dir}\n")
            f.write(f"Bounding box data: {self.bbox_dir}\n")
            f.write(f"Plots: {self.plots_dir}\n\n")
            
            f.write("DISEASE CATEGORIES\n")
            f.write("-" * 30 + "\n")
            for cat in self.disease_categories:
                f.write(f"- {cat}\n")
            
            f.write("\nFILES CREATED\n")
            f.write("-" * 30 + "\n")
            f.write("- data/processed/classification/labels.csv\n")
            f.write("- data/processed/bbox/annotations_coco.json\n")
            f.write("- data/processed/bbox/bbox_original.csv\n")
            f.write("- data/plots/*.png (visualization plots)\n")
            f.write("- data/dataset_summary.txt (this file)\n")
        
        print(f"\n📄 Summary report saved to: {report_path}")
    
    def run(self):
        """Main function to organize the entire dataset."""
        print("🚀 Starting NIH Chest X-ray dataset organization...")
        
        # Step 1: Download if needed
        self.download_dataset()
        
        # Step 2: Load metadata
        print("\n📊 Loading metadata...")
        entry_df, bbox_df = self.load_metadata()
        print(f"   Total images: {len(entry_df):,}")
        print(f"   Images with bboxes: {len(bbox_df['Image Index'].unique()):,}")
        
        # Step 3: Generate visualizations
        print("\n📈 Generating visualizations...")
        self.plot_disease_distribution(entry_df)
        self.plot_patient_demographics(entry_df)
        self.plot_multi_label_statistics(entry_df)
        print(f"   Plots saved to: {self.plots_dir}")
        
        # Step 4: Get bbox filenames
        bbox_filenames = set(bbox_df['Image Index'])
        
        # Step 5: Organize classification data
        print("\n📁 Organizing classification data...")
        class_df = self.organize_classification_data(entry_df, bbox_filenames)
        
        # Step 6: Organize bbox data with COCO format
        print("\n📁 Organizing bbox data...")
        coco_data = self.organize_bbox_data(bbox_df, entry_df)
        
        # Step 7: Generate additional plots
        self.plot_bbox_vs_classification_split(len(class_df), len(coco_data['images']))
        self.plot_bbox_disease_distribution(coco_data)
        
        # Step 8: Generate summary report
        self.generate_summary_report(entry_df, bbox_df, class_df, coco_data)
        
        print("\n✅ Dataset organization complete!")
        print(f"\n📊 Final Statistics:")
        print(f"   Classification images: {len(class_df):,}")
        print(f"   Bbox images: {len(coco_data['images']):,}")
        print(f"   Total annotations: {len(coco_data['annotations']):,}")
        
        # Print folder structure
        print(f"\n📁 Data Structure:")
        print(f"   data/")
        print(f"   ├── raw/")
        print(f"   │   └── nih-chest-xrays/")
        print(f"   ├── processed/")
        print(f"   │   ├── classification/")
        print(f"   │   │   ├── images/")
        print(f"   │   │   └── labels.csv")
        print(f"   │   └── bbox/")
        print(f"   │       ├── images/")
        print(f"   │       ├── annotations_coco.json")
        print(f"   │       └── bbox_original.csv")
        print(f"   ├── plots/")
        print(f"   │   ├── disease_distribution.png")
        print(f"   │   ├── dataset_split.png")
        print(f"   │   ├── bbox_disease_distribution.png")
        print(f"   │   ├── patient_demographics.png")
        print(f"   │   └── multi_label_distribution.png")
        print(f"   └── dataset_summary.txt")


def main():
    """Example usage."""
    organizer = NIHDatasetOrganizer()
    organizer.run()


if __name__ == "__main__":
    main()
