
# Safety Tracking Inference

This repository contains the codebase for Safety Tracking Inference on the MMDET framework. The project focuses on object detection and tracking to ensure safety compliance by monitoring various safety equipment and procedures in different environments.

## TO DO
- [X] REDO create_env script, and clean up redundancy with configuration function
- [ ] Hard code models based on service, we should not be able to run different models with different logic based on the run script.


## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Docker](#docker)
- [Configuration](#configuration)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)

## Project Description

The Safety Tracking Inference project leverages state-of-the-art machine learning algorithms to detect and track safety equipment such as fire extinguishers, safety signs, and other critical safety apparatus. The goal is to enhance safety compliance and ensure all equipment is in place and operational.

## Features

- Object Detection: Detects various safety equipment using pre-trained models.
- Tracking: Tracks the detected objects across video frames to monitor their status.
- Compliance Verification: Verifies the presence and condition of safety equipment.
- Modular Design: Easy to extend and customize for different safety compliance scenarios.

## Installation

To set up the project locally, follow these steps:

### System Preparation

### Build and Run Docker Container

- **Build the Docker image:**
    ```sh
    docker build -f Dockerfile -t safety-tracking-inference:gh.1.1 --build-arg GIT_USERNAME=GIT_USERNAME --build-arg GIT_PAT=GIT_PAT --build-arg AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID --build-arg AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY --build-arg AWS_DEFAULT_REGION=AWS_DEFAULT_REGION --build-arg SAFETY_SET=gh --build-arg VERSION=1.2 .
    ```

- **Launch the Docker container:**
  ```sh
  docker run --shm-size=8g -it --gpus all safety-tracking-inference:gh.1.1
  ```

1. **Update and install necessary packages:**
   ```bash
   sudo apt update
   sudo apt install python3-pip
   python3 -m pip install --upgrade pip
   sudo apt-get install protobuf-compiler libprotobuf-dev
   ```

2. **Install CPU monitoring tools:**
   ```bash
   sudo apt install snapd
   sudo snap install bpytop
   bpytop
   ```

3. **Install GPU monitoring tools:**
   ```bash
   sudo apt install pipx
   sudo apt install python3.8-venv
   pipx run nvitop
   ```

### AWS ECR - Uploading Docker Images

To upload a Docker image to AWS ECR, follow these steps:

1. **Set Up AWS CLI:**
   Ensure AWS CLI is installed on your machine.

2. **Configure AWS CLI:**
   Use `aws configure` to set up your credentials.

3. **Authenticate Docker to ECR:**
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 521337707473.dkr.ecr.us-east-1.amazonaws.com
   ```

4. **Tag the Docker Image with both Version and Latest:**
   ```bash
   docker tag safety-tracking-inference:[version] [your-aws-account-id].dkr.ecr.[region].amazonaws.com/safety-tracking-inference:[version]
   ```

5. **Push the Image:**
   ```bash
   docker push [your-aws-account-id].dkr.ecr.[region].amazonaws.com/safety-tracking-inference:[version]
   ```

### Clone Repo and Set Up Environment

4. **Clone the repository and navigate into it:**
   ```bash
   git clone https://${GIT_USERNAME}:${GIT_PAT}@github.com/nexterarobotics/safety-tracking-inference.git
   cd safety-tracking-inference
   ```

5. **Set up a virtual environment:**
   ```bash
   python3 -m venv safety
   source safety/bin/activate
   python3 -m pip install --upgrade pip
   pip install --upgrade pip
   ```

6. **Install PyTorch and MMCV:**
   - For CUDA 12.1:
     ```bash
     pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
     pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.10/index.html
     ```
   - For CUDA 11.8:
     ```bash
     pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
     pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.10/index.html
     ```
   - For CUDA 11.3:
     ```bash
     pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
     pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
     ```

7. **Install project requirements:**
   ```bash
   pip install -r requirements.txt
   ```

### AWS Configuration

8. **Configure AWS CLI:**
   ```bash
   pip install --upgrade awscli
   aws configure
   ```

9. **Additional configuration:**
   ```bash
   export PATH=$PATH:/snap/bin
   conda config --set auto_activate_base false
   # IF ERROR: Failed to initialize NumPy: _ARRAY_API not found
   pip install --force-reinstall -v "numpy==1.25.2"
   ```

## Testing

### Step 1: Download Pretrained Weights from AWS

Download the necessary pretrained weights for inference.

```bash
aws s3 cp s3://didge-cv-models/Co-DETR/gh_1.2/best_bbox_mAP.pth gh/
aws s3 cp s3://didge-cv-models/Co-DETR/ppe_1.1/best_bbox_mAP.pth ppe/
aws s3 cp s3://didge-cv-models/Co-DETR/hlr_1.0/best_bbox_mAP.pth hlr/
aws s3 cp s3://didge-cv-models/Co-DETR/sse_1.0/best_bbox_mAP.pth sse/
```

### Step 2: Download Validation JSON Files

Fetch the validation JSON files for each safety set.

```bash
aws s3 cp s3://didge-cv-models/Co-DETR/gh_1.2/gh_val.json data/annotations/
aws s3 cp s3://didge-cv-models/Co-DETR/ppe_1.1/ppe_val.json data/annotations/
aws s3 cp s3://didge-cv-models/Co-DETR/hlr_1.0/hlr_val.json data/annotations/
aws s3 cp s3://didge-cv-models/Co-DETR/sse_1.0/sse_val.json data/annotations/
```

### Step 3: Download Images

Sync the image data for each safety set.

```bash
aws s3 sync s3://didge-cv-annotation-data/safety-tracking/gh data/annotations/gh
aws s3 sync s3://didge-cv-annotation-data/safety-tracking/ppe data/annotations/ppe
aws s3 sync s3://didge-cv-annotation-data/safety-tracking/hlr data/annotations/hlr
aws s3 sync s3://didge-cv-annotation-data/safety-tracking/sse data/annotations/sse
```

### Step 4: Move Images

Organize the images into the appropriate directories.

```bash
python3 projects/misc/move_images.py --safety_set gh
python3 projects/misc/move_images.py --safety_set ppe
python3 projects/misc/move_images.py --safety_set hlr
python3 projects/misc/move_images.py --safety_set sse
```

### Step 5: Run Inference Script

Perform inference on the downloaded datasets.

```bash
python3 inference.py --safety-model gh
python3 inference.py --safety-model ppe
python3 inference.py --safety-model hlr
python3 inference.py --safety-model sse
```

### Step 6: Run Test Script

Validate the model performance with the test script.

```bash
python3 test.py --safety_set gh
python3 test.py --safety_set ppe
python3 test.py --safety_set hlr
python3 test.py --safety_set sse
```

## Configuration

Configuration files are located in the `configs/` directory. You can adjust model parameters, inference settings, and other options by editing these configuration files.

## Models

The models used in this project are based on the MMDetection framework. Pre-trained models can be found in the `models/` directory. You can also train your own models using the provided training scripts.
