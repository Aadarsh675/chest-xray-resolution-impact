# ----------------------------------------
# Build & Run Instructions
# ----------------------------------------
    
# Build: 
    # docker build --no-cache -f Dockerfile -t safety-tracking-inference:gh-v1.2  \
    # --build-arg GIT_USERNAME= \
    # --build-arg GIT_PAT= \
    # --build-arg AWS_ACCESS_KEY_ID= \
    # --build-arg AWS_SECRET_ACCESS_KEY= \
    # --build-arg AWS_DEFAULT_REGION= \
    # --build-arg NEXTERA_API_HOST= \
    # --build-arg NEXTERA_API_KEY= \
    # --build-arg REDIS_DB_URL= \
    # --build-arg REDIS_DB_PASSWORD= \
    # --build-arg REDIS_DB_PORT= \
    # --build-arg RABIT_MQ_URL= \
    # --build-arg SAFETY_SET=gh \
    # --build-arg VERSION=1.2 \
    # .
# ----------------------------------------

# Run: 
    # docker run --shm-size=8g -it \
	# -e SAFETY_MODEL=gh \
	# --gpus all \
	# safety-tracking-inference:gh-v1.2
# ----------------------------------------

# Base Image
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# Setup Timezone
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Setup Non-Interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nvidia-container-toolkit \
    python3 python3-pip \
    git wget cmake \
    libsm6 libxext6 libx11-dev libgl1-mesa-glx \
    libxxf86vm-dev libxcursor-dev libxi-dev \
    libxrandr-dev libxinerama-dev libglew-dev \
    libxrender1 libxkbcommon-x11-0 libxkbcommon0 parallel \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Model and Version Arguments
ARG VERSION
ARG SAFETY_SET

# GitHub Credentials
ARG GIT_PAT
ARG GIT_USERNAME

# AWS Credentials
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION 
ARG AWS_SECRET_ACCESS_KEY
ARG DEFAULT_S3_BUCKET

# Nextera Robotics Credentials
ARG NEXTERA_API_HOST
ARG NEXTERA_API_KEY 

# Redis Credentials
ARG REDIS_DB_URL
ARG REDIS_DB_PASSWORD
ARG REDIS_DB_PORT

# RabbitMQ URL
ARG RABIT_MQ_URL

# Download the code from GitHub
RUN git clone https://${GIT_USERNAME}:${GIT_PAT}@github.com/nexterarobotics/safety-tracking-inference.git \
    && cd safety-tracking-inference

# Install PyTorch and MMCV
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html 
RUN pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# Install Requirements
RUN cd safety-tracking-inference && \
    git pull && \
    pip install -r requirements.txt

# Go back to the root directory
WORKDIR /

# Application Code & Blender Setup
RUN mv safety-tracking-inference /app

# Enviorment Variables for the Application
ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1

# Environment Variables for the Application
ENV NEXTERA_API_HOST=${NEXTERA_API_HOST}
ENV NEXTERA_API_KEY=${NEXTERA_API_KEY}
ENV REDIS_DB_URL=${REDIS_DB_URL}
ENV REDIS_DB_PASSWORD=${REDIS_DB_PASSWORD}
ENV REDIS_DB_PORT=${REDIS_DB_PORT}
ENV RABIT_MQ_URL=${RABIT_MQ_URL}
ENV DEFAULT_S3_BUCKET=${DEFAULT_S3_BUCKET}

# AWS Configuration
RUN pip install --upgrade awscli
RUN aws configure set aws_access_key_id ${AWS_ACCESS_KEY_ID} && \
    aws configure set aws_secret_access_key ${AWS_SECRET_ACCESS_KEY} && \
    aws configure set default.region ${AWS_DEFAULT_REGION} && \
    aws configure set default.output json

# Download the pretrained weights from S3
RUN mkdir -p /app/${SAFETY_SET}
RUN aws s3 cp s3://didge-cv-models/Co-DETR/${SAFETY_SET}_${VERSION}/best_bbox_mAP.pth /app/${SAFETY_SET}

# Set the working directory for the container
WORKDIR /app

# Ensure run.sh is present and executable, then execute it on container start
RUN chmod +x /app/run.sh
CMD ["bash", "run.sh"]