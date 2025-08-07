import pandas as pd
from google.colab import drive
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

# Mount Google Drive
drive.mount('/content/drive')

# Specify the path to the CSV file and image directory on Google Drive
csv_path = '/content/drive/My Drive/nih_chest_xray_dataset/BBox_List_2017.csv'  # Adjust path as needed
image_dir = '/content/drive/My Drive/nih_chest_xray_dataset/images'  # Adjust to your image directory

# Load the CSV data into a Pandas DataFrame
df = pd.read_csv(csv_path)

# Display the number of unique images in the dataset
num_images = df['Image Index'].nunique()
print(f"Number of unique images in the dataset: {num_images}")

# Function to parse bounding box coordinates from the 'Bbox [x,y,w,h]' column
def parse_bbox(bbox_str):
    # Remove brackets and split by comma
    bbox = bbox_str.strip('[]').split(',')
    return [float(coord) for coord in bbox]

# Select a few example images (e.g., first 3)
num_examples = 3
example_images = df.head(num_examples)

# Create a figure for example image visualization
plt.figure(figsize=(15, 5 * num_examples))

for i, row in example_images.iterrows():
    # Get image path
    image_path = os.path.join(image_dir, row['Image Index'])
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found, skipping...")
        continue
    
    # Load image
    img = Image.open(image_path)
    
    # Parse bounding box coordinates
    x, y, w, h = parse_bbox(row['Bbox [x,y,w,h]'])
    
    # Create subplot
    plt.subplot(num_examples, 1, i + 1)
    plt.imshow(img, cmap='gray')  # Assuming medical images are grayscale
    
    # Create a rectangle patch for the bounding box
    rect = patches.Rectangle(
        (x, y), w, h,
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )
    
    # Add the rectangle to the plot
    plt.gca().add_patch(rect)
    
    # Add title with image index and finding label
    plt.title(f"Image: {row['Image Index']} | Label: {row['Finding Label']}")
    
    # Remove axes for cleaner visualization
    plt.axis('off')

# Adjust layout and display example images
plt.tight_layout()
plt.show()

# Create a pie chart of images by their Finding Label
# Group by Image Index and Finding Label to get unique images per label
label_counts = df.groupby('Image Index')['Finding Label'].first().value_counts()

# Create a new figure for the pie chart
plt.figure(figsize=(10, 8))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Images by Disease Label')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular

# Display the pie chart
plt.show()
