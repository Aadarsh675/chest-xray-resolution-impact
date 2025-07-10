{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# NIH Chest X-ray Pipeline Notebook\n",
    "\n",
    "This notebook provides an interactive way to run the NIH Chest X-ray dataset pipeline."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup Environment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check if running in Google Colab\n",
    "import os\n",
    "IN_COLAB = 'COLAB_GPU' in os.environ\n",
    "\n",
    "if IN_COLAB:\n",
    "    print(\"Running in Google Colab\")\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive')\n",
    "else:\n",
    "    print(\"Running in local environment\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install required packages\n",
    "!pip install kagglehub pandas numpy scikit-learn Pillow matplotlib seaborn tqdm\n",
    "!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n",
    "!pip install ultralytics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create necessary files\n",
    "# Note: In a real scenario, you would have these files already\n",
    "# This cell creates them inline for demonstration\n",
    "\n",
    "files_to_create = [\n",
    "    'main.py',\n",
    "    'dataset_downloader.py', \n",
    "    'gdrive_handler.py',\n",
    "    'data_formatter.py',\n",
    "    'model_trainer.py'\n",
    "]\n",
    "\n",
    "print(\"Pipeline files ready!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Configure Pipeline Parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Configuration\n",
    "config = {\n",
    "    'data_dir': './data',\n",
    "    'gdrive_dir': '/content/drive/MyDrive/nih-chest-xrays' if IN_COLAB else '',\n",
    "    'epochs': 5,  # Reduced for demo\n",
    "    'batch_size': 16,\n",
    "    'model': 'yolov8n',  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x\n",
    "    'train_split': 0.8,\n",
    "    'val_split': 0.1\n",
    "}\n",
    "\n",
    "print(\"Configuration:\")\n",
    "for key, value in config.items():\n",
    "    print(f\"  {key}: {value}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Set up Kaggle Credentials (if needed)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Option 1: Upload kaggle.json file\n",
    "# Uncomment if you have a kaggle.json file to upload\n",
    "\n",
    "# from google.colab import files\n",
    "# uploaded = files.upload()\n",
    "# !mkdir -p ~/.kaggle\n",
    "# !cp kaggle.json ~/.kaggle/\n",
    "# !chmod 600 ~/.kaggle/kaggle.json"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Option 2: Set credentials manually\n",
    "# Uncomment and fill in your credentials\n",
    "\n",
    "# import os\n",
    "# os.environ['KAGGLE_USERNAME'] = 'your_username'\n",
    "# os.environ['KAGGLE_KEY'] = 'your_api_key'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Run Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run the main pipeline\n",
    "!python main.py \\\n",
    "    --data-dir {config['data_dir']} \\\n",
    "    --gdrive-dir \"{config['gdrive_dir']}\" \\\n",
    "    --epochs {config['epochs']} \\\n",
    "    --batch-size {config['batch_size']} \\\n",
    "    --model {config['model']} \\\n",
    "    --train-split {config['train_split']} \\\n",
    "    --val-split {config['val_split']}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Analyze Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "\n",
    "# Load training results\n",
    "with open('./data/results/training_results.json', 'r') as f:\n",
    "    train_results = json.load(f)\n",
    "\n",
    "print(\"Training Results:\")\n",
    "print(f\"Model: {train_results['model']}\")\n",
    "print(f\"Epochs: {train_results['epochs']}\")\n",
    "print(f\"Device: {train_results['device']}\")\n",
    "print(\"\\nMetrics:\")\n",
    "for metric, value in train_results['metrics'].items():\n",
    "    print(f\"  {metric}: {value:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load and display dataset statistics\n",
    "with open('./data/coco_format/dataset_stats.json', 'r') as f:\n",
    "    stats = json.load(f)\n",
    "\n",
    "# Create visualization\n",
    "categories = list(stats['categories'].keys())\n",
    "counts = list(stats['categories'].values())\n",
    "\n",
    "plt.figure(figsize=(12, 6))\n",
    "plt.bar(categories, counts)\n",
    "plt.xticks(rotation=45, ha='right')\n",
    "plt.title('Distribution of Disease Classes in Dataset')\n",
    "plt.xlabel('Disease Class')\n",
    "plt.ylabel('Number of Annotations')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(f\"\\nTotal images: {stats['total_images']}\")\n",
    "print(f\"Total annotations: {stats['total_annotations']}\")\n",
    "print(f\"Bounding box annotations: {stats['bbox_annotations']}\")\n",
    "print(f\"Full image annotations: {stats['full_image_annotations']}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display sample predictions\n",
    "from IPython.display import Image, display\n",
    "import glob\n",
    "\n",
    "# Find visualization images\n",
    "vis_images = glob.glob('./data/visualizations/test_prediction_*.png')\n",
    "\n",
    "if vis_images:\n",
    "    print(\"Sample Test Predictions:\")\n",
    "    for img_path in vis_images[:3]:  # Show first 3\n",
    "        display(Image(img_path, width=600))\n",
    "else:\n",
    "    print(\"No visualization images found\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Use Trained Model for Inference"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from ultralytics import YOLO\n",
    "import glob\n",
    "\n",
    "# Find the trained model\n",
    "model_paths = glob.glob('./data/runs/train_*/weights/best.pt')\n",
    "if model_paths:\n",
    "    model_path = model_paths[-1]  # Use most recent\n",
    "    print(f\"Loading model from: {model_path}\")\n",
    "    \n",
    "    # Load model\n",
    "    model = YOLO(model_path)\n",
    "    \n",
    "    # Example inference on a test image\n",
    "    test_images = glob.glob('./data/coco_format/images/*.png')[:5]\n",
    "    \n",
    "    for img_path in test_images[:1]:  # Run on first image\n",
    "        results = model(img_path, conf=0.25)\n",
    "        \n",
    "        # Display results\n",
    "        for r in results:\n",
    "            im_array = r.plot()  # plot a BGR numpy array of predictions\n",
    "            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image\n",
    "            display(im)\n",
    "else:\n",
    "    print(\"No trained model found\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Save Model and Results to Google Drive"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "if IN_COLAB:\n",
    "    import shutil\n",
    "    from datetime import datetime\n",
    "    \n",
    "    # Create a timestamp\n",
    "    timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n",
    "    \n",
    "    # Define save path\n",
    "    save_path = f\"/content/drive/MyDrive/nih-chest-xrays/models/run_{timestamp}\"\n",
    "    os.makedirs(save_path, exist_ok=True)\n",
    "    \n",
    "    # Copy model and results\n",
    "    if model_paths:\n",
    "        shutil.copy2(model_paths[-1], save_path)\n",
    "        print(f\"Model saved to: {save_path}\")\n",
    "    \n",
    "    # Copy results\n",
    "    shutil.copytree('./data/results', f\"{save_path}/results\", dirs_exist_ok=True)\n",
    "    print(f\"Results saved to: {save_path}/results\")\n",
    "else:\n",
    "    print(\"Not in Colab, skipping Google Drive save\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Clean Up (Optional)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Uncomment to clean up local data (keeps Google Drive data)\n",
    "# !rm -rf ./data/nih-chest-xrays  # Remove downloaded dataset\n",
    "# !rm -rf ./data/coco_format/images  # Remove processed images\n",
    "# print(\"Cleaned up local data\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}