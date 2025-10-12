import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.environment import get_environment_config

# Environment setup
config = get_environment_config()

def load_coco_annotations(file_path):
    """Load COCO annotations with error handling"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"[WARN] COCO file not found: {file_path}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load COCO file {file_path}: {e}")
        return None

def get_category_distribution(data):
    """Get category distribution from COCO data"""
    if data is None:
        return [], []
    
    # Map category IDs to names
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Count occurrences of each category in annotations
    category_counts = Counter()
    for ann in data['annotations']:
        category_counts[ann['category_id']] += 1
    
    # Sort by category ID for consistent ordering
    sorted_ids = sorted(category_counts.keys())
    labels = [cat_id_to_name[cat_id] for cat_id in sorted_ids]
    counts = [category_counts[cat_id] for cat_id in sorted_ids]
    
    # Normalize the counts
    total = sum(counts)
    norm_counts = [count / total for count in counts] if total > 0 else counts
    
    return labels, norm_counts

def plot_normalized_histogram(labels, norm_counts, title):
    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))
    plt.bar(x, norm_counts, align='center')
    plt.xticks(x, labels, rotation=90)
    plt.xlabel('Categories')
    plt.ylabel('Normalized Frequency')
    plt.title(title)
    plt.tight_layout()
    
    # Save plot instead of showing
    plots_dir = os.path.join(config['data_root'], 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_filename = f"{title.lower().replace(' ', '_')}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()  # Close the figure to free memory

def main():
    """Main function to visualize split data"""
    # Use environment-aware paths
    test_file = os.path.join(config['anno_dir'], 'test_annotations_coco.json')
    train_file = os.path.join(config['anno_dir'], 'train_annotations_coco.json')
    
    print(f"Looking for COCO files:")
    print(f"  Test: {test_file}")
    print(f"  Train: {train_file}")
    
    test_data = load_coco_annotations(test_file)
    train_data = load_coco_annotations(train_file)
    
    if test_data is None and train_data is None:
        print("\n[ERROR] No COCO annotation files found!")
        print("Please run the COCO conversion first:")
        print("  python src/coco_converter.py")
        return
    
    # Process test data
    if test_data is not None:
        test_labels, test_norm = get_category_distribution(test_data)
        if test_labels:
            plot_normalized_histogram(test_labels, test_norm, 'Normalized Histogram for Test Annotations')
        else:
            print("[WARN] No test data to visualize")
    
    # Process train data
    if train_data is not None:
        train_labels, train_norm = get_category_distribution(train_data)
        if train_labels:
            plot_normalized_histogram(train_labels, train_norm, 'Normalized Histogram for Train Annotations')
        else:
            print("[WARN] No train data to visualize")

if __name__ == "__main__":
    main()
