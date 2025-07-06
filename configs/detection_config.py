#!/usr/bin/env python
#-*- coding: utf-8 -*-
#
# ==============================================================================
# Configuration module for inference
#
# Created on: Thu Mar 07 2024
# ==============================================================================

import os
import boto3
from botocore.exceptions import ClientError
from pathlib import Path

def setup_paths(args):
    """
    Set up the configuration paths based on model type

    Args:
        args: Command line arguments
    """

    # Set the configuration paths for codetr model
    if args.model == 'codetr':
        args.config = f'services/detection/mmdet/configs/{args.model}_{args.dataset}/codetr_config.py'
        args.checkpoint = f'weights/{args.model}/{args.dataset}/best_model.pth'

    # Download weights if they don't exist locally
    args = download_weights(args)

    return args

def download_weights(args):
    """
    Check if weights exist at the expected path, and download from S3 if not.

    Args:
        args: Command line arguments containing model and dataset info
    """

    # Define S3 path for codetr model
    s3_paths = {
        ('codetr', 'nih-chest-xray'): 's3://medical-cv-models/codetr/nih-chest-xray/best_model.pth',
    }

    # Check if weights file already exists
    if os.path.exists(args.checkpoint):
        print(f"Weights file already exists at {args.checkpoint}")
        return args

    # Get S3 path for the requested model/dataset
    key = (args.model, args.dataset)
    if key not in s3_paths:
        raise ValueError(f"No S3 path defined for model {args.model} and dataset {args.dataset}")

    s3_path = s3_paths[key]
    bucket_name = s3_path.split('/')[2]
    object_key = '/'.join(s3_path.split('/')[3:])

    # Create directory if it doesn't exist
    checkpoint_dir = os.path.dirname(args.checkpoint)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Download file from S3
    try:
        print(f"Downloading weights from {s3_path} to {args.checkpoint}...")
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket_name, object_key, args.checkpoint)
        print(f"Successfully downloaded weights to {args.checkpoint}")
    except ClientError as e:
        print(f"Error downloading weights: {e}")
        raise

    return args

def get_gpu_config(model, dataset):
    """
    Get GPU configuration settings based on model and dataset combination

    Args:
        model: Model name
        dataset: Dataset name

    Returns:
        Dictionary containing GPU configuration settings
    """
    # Configuration for codetr model
    gpu_config = {
        'gpu_multiple': 2,  # Number of worker processes per GPU
        'timeout': 6    # Timeout in seconds
    }

    # Model-dataset specific configurations
    config_map = {
        ('codetr', 'nih-chest-xray'): {'gpu_multiple': 2, 'timeout': 6},
    }

    # Get specific configuration if available
    key = (model, dataset)
    if key in config_map:
        gpu_config.update(config_map[key])
    else:
        raise ValueError(f"No configuration defined for model {model} and dataset {dataset}")

    return gpu_config

def get_config_paths(model, dataset):
    """
    Get configuration paths for the specified model and dataset

    Args:
        model: Model name
        dataset: Dataset name

    Returns:
        Dictionary of configuration paths
    """
    # Define base path for configurations
    base_path = f"services/detection/mmdet/configs/{model}_{dataset}"

    # Initialize with the paths that are always needed
    config_paths = {
        'db_id': f"{base_path}/db_id.json",
        'class_color': f"{base_path}/class_color.txt"
    }

    return config_paths