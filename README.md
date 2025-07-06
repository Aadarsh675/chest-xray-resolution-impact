# chest-xray-resolution-impact
A study on how image resolution affects state-of-the-art bounding box detection models on the NIH Chest X-ray dataset.

## Project Structure
```
chest-xray-resolution-impact/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                    # Original NIH dataset (not tracked by git)
│   │   ├── images_001/
│   │   ├── images_002/
│   │   ├── ...
│   │   ├── Data_Entry_2017.csv
│   │   └── BBox_List_2017.csv
│   │
│   └── processed/              # Resized images for experiments
│       ├── resolution_256/
│       │   └── images/
│       ├── resolution_512/
│       │   └── images/
│       ├── resolution_768/
│       │   └── images/
│       └── resolution_1024/    # Original resolution
│           └── images/
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Resize images to different resolutions
│   ├── dataset.py             # PyTorch dataset class
│   ├── model.py               # SOTA model wrapper (e.g., YOLO, Faster R-CNN)
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation metrics
│   └── utils.py               # Helper functions
│
├── configs/
│   ├── base_config.yaml       # Base configuration
│   ├── res_256_config.yaml    # Config for 256x256 training
│   ├── res_512_config.yaml    # Config for 512x512 training
│   ├── res_768_config.yaml    # Config for 768x768 training
│   └── res_1024_config.yaml   # Config for 1024x1024 training
│
└── experiments/
    ├── logs/                  # Training logs
    ├── checkpoints/           # Model checkpoints
    └── results/               # Evaluation results
```
    
## Setup
Clone the repository
bash
git clone https://github.com/yourusername/chest-xray-resolution-impact.git
1. cd chest-xray-resolution-impact
2. Install dependencies
- bash
- pip install -r requirements.txt
3. Download NIH Chest X-ray Dataset
- Download the dataset from Kaggle NIH Chest X-rays
- Extract all zip files into data/raw/
4. Prepare multi-resolution datasets
- bash
- python src/data_preprocessing.py --input_dir data/raw --output_dir data/processed


## Usage
### Train model on specific resolution:
bash<br/>
python src/train.py --config configs/res_512_config.yaml
### Evaluate all models:
bash<br/>
python src/evaluate.py --checkpoint_dir experiments/checkpoints

## Requirements.txt Example
- txt
- torch>=2.0.0
- torchvision>=0.15.0
- opencv-python>=4.8.0
- pandas>=2.0.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- pyyaml>=6.0
- tqdm>=4.65.0
- albumentations>=1.3.0
- tensorboard>=2.13.0

## Data Format
The processed data maintains the same structure but with resized images:
- Each resolution folder contains the same images resized accordingly
- Bounding box coordinates are scaled proportionally
- Metadata CSVs are copied to each resolution folder with adjusted coordinates

## Notes
- Original 1024x1024 images are copied to resolution_1024/ without modification
- Bounding box annotations are only available for a subset of images
- The .gitignore should exclude the data/ folder due to large file sizes
