# NIH Chest X-ray Dataset Pipeline

A complete pipeline for downloading, organizing, and training object detection models on the NIH Chest X-ray dataset in COCO format.

## Features

- **Automatic dataset management**: Downloads NIH Chest X-ray dataset using kagglehub
- **Google Drive integration**: Stores and retrieves dataset from Google Drive to avoid re-downloading
- **COCO format conversion**: Converts NIH dataset to COCO format for compatibility with modern object detection frameworks
- **Multi-framework support**: Works with Ultralytics YOLO and Detectron2
- **Automatic data splitting**: Creates train/validation/test splits
- **Comprehensive evaluation**: Includes testing and visualization capabilities

## Installation

1. Clone this repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. For Google Colab users:
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Usage

### Basic Usage

Run the complete pipeline with default settings:

```bash
python main.py
```

### Advanced Usage

Customize the pipeline with command-line arguments:

```bash
python main.py \
    --data-dir ./my_data \
    --gdrive-dir /content/drive/MyDrive/nih-xrays \
    --train-split 0.7 \
    --val-split 0.15 \
    --epochs 20 \
    --batch-size 16 \
    --model yolov8m
```

### Command-line Arguments

- `--data-dir`: Local directory for data storage (default: ./data)
- `--gdrive-dir`: Google Drive directory path (default: /content/drive/MyDrive/nih-chest-xrays)
- `--train-split`: Training set ratio (default: 0.8)
- `--val-split`: Validation set ratio (default: 0.1)
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 32)
- `--model`: Model to use - yolov8n, yolov8s, yolov8m, yolov8l, yolov8x (default: yolov8n)

## Pipeline Steps

1. **Dataset Check**: Checks if dataset exists in Google Drive
2. **Download**: Downloads dataset using kagglehub if not found
3. **Upload to Drive**: Uploads dataset to Google Drive for future use
4. **COCO Conversion**: Converts NIH format to COCO format
5. **Data Splitting**: Creates train/val/test splits
6. **Model Training**: Trains object detection model
7. **Evaluation**: Tests model on test set
8. **Visualization**: Creates prediction visualizations

## Dataset Structure

After processing, the dataset will be organized as follows:

```
data/
├── nih-chest-xrays/          # Original dataset
│   ├── Data_Entry_2017_v2020.csv
│   ├── BBox_List_2017.csv
│   └── images/
├── coco_format/              # COCO format dataset
│   ├── images/               # All images
│   ├── annotations/          # COCO annotations
│   │   ├── instances_all.json
│   │   ├── instances_train.json
│   │   ├── instances_val.json
│   │   └── instances_test.json
│   ├── data.yaml            # YOLO config
│   └── dataset_stats.json   # Dataset statistics
├── runs/                    # Training runs
└── results/                 # Final results
```

## Disease Classes

The pipeline handles 15 disease classes from the NIH dataset:
- Atelectasis
- Cardiomegaly
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Consolidation
- Edema
- Emphysema
- Fibrosis
- Pleural_Thickening
- Hernia
- No Finding

## Output Files

After running the pipeline, you'll find:

- **Training results**: `results/training_results.json`
- **Test results**: `results/test_results.json`
- **Model weights**: `data/runs/train_*/weights/best.pt`
- **Visualizations**: `data/visualizations/`
- **Dataset statistics**: `data/coco_format/dataset_stats.json`

## Google Colab Example

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository (if needed)
!git clone https://github.com/your-repo/nih-xray-pipeline.git
%cd nih-xray-pipeline

# Install requirements
!pip install -r requirements.txt

# Run pipeline
!python main.py --epochs 5 --batch-size 16
```

## Troubleshooting

### Kaggle API Issues
If you encounter Kaggle API issues:
1. Ensure you have a Kaggle account
2. Create an API token from Kaggle settings
3. Set up authentication:
```python
import os
os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY'] = 'your_api_key'
```

### Memory Issues
For large batch sizes or high-resolution training:
- Reduce batch size: `--batch-size 8`
- Use smaller model: `--model yolov8n`
- Enable gradient checkpointing if available

### Google Drive Space
The NIH dataset is large (~42GB). Ensure you have sufficient Google Drive storage.

## Custom Models

To use a custom model, modify the `ModelTrainer` class in `model_trainer.py`:

```python
# Example: Adding a custom model
if self.model_name == 'custom_model':
    self.model = load_custom_model()
```

## Citation

If you use this pipeline, please cite the NIH Chest X-ray dataset:

```
Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. 
ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on 
Weakly-Supervised Classification and Localization of Common Thorax Diseases. 
IEEE CVPR 2017
```

## License

This pipeline is provided as-is for research purposes. The NIH Chest X-ray dataset has its own license terms - please review them before use.
