import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def load_coco_annotations(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_category_distribution(data):
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
    plt.show()

# Main execution
test_file = 'data/annotations/test_annotations_coco.json'
train_file = 'data/annotations/train_annotations_coco.json'

test_data = load_coco_annotations(test_file)
train_data = load_coco_annotations(train_file)

test_labels, test_norm = get_category_distribution(test_data)
train_labels, train_norm = get_category_distribution(train_data)

# Assuming categories are the same in both, but if not, this might need adjustment
plot_normalized_histogram(test_labels, test_norm, 'Normalized Histogram for Test Annotations')
plot_normalized_histogram(train_labels, train_norm, 'Normalized Histogram for Train Annotations')
