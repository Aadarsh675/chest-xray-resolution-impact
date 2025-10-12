"""
Environment detection and configuration utilities.

This module provides environment-aware path configuration for the chest X-ray
resolution impact project, automatically detecting whether the code is running
on Google Colab or a local machine and setting appropriate paths.
"""

import os
import sys
from typing import Dict, Any


def is_colab() -> bool:
    """
    Check if running in Google Colab environment.
    
    Returns:
        bool: True if running in Colab, False otherwise
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def is_jupyter() -> bool:
    """
    Check if running in Jupyter notebook environment.
    
    Returns:
        bool: True if running in Jupyter, False otherwise
    """
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def get_data_root() -> str:
    """
    Get the data root directory based on environment.
    
    Returns:
        str: Path to data root directory
    """
    if is_colab():
        return '/content'
    else:
        # Local environment - use data subdirectory
        data_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_dir, exist_ok=True)
        return data_dir


def get_image_root() -> str:
    """
    Get the image root directory based on environment.
    
    Returns:
        str: Path to image root directory
    """
    if is_colab():
        return '/content/drive/MyDrive/vindr_pcxr'
    else:
        # Check for mounted drive first
        mounted_drive_path = r"G:\My Drive\vindr_pcxr"
        if os.path.exists(mounted_drive_path):
            return mounted_drive_path
        else:
            # Fallback to local data directory
            data_dir = os.path.join(os.getcwd(), 'data')
            os.makedirs(data_dir, exist_ok=True)
            return data_dir


def get_weights_dir() -> str:
    """
    Get the weights directory based on environment.
    
    Returns:
        str: Path to weights directory
    """
    if is_colab():
        return '/content/drive/MyDrive/weights'
    else:
        weights_dir = os.path.join(os.getcwd(), 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        return weights_dir


def get_analysis_dir() -> str:
    """
    Get the analysis directory based on environment.
    
    Returns:
        str: Path to analysis directory
    """
    if is_colab():
        return '/content/analysis'
    else:
        analysis_dir = os.path.join(os.getcwd(), 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        return analysis_dir


def get_anno_dir() -> str:
    """
    Get the annotations directory based on environment.
    
    Returns:
        str: Path to annotations directory
    """
    if is_colab():
        return '/content/annotations'
    else:
        anno_dir = os.path.join(os.getcwd(), 'data', 'annotations')
        os.makedirs(anno_dir, exist_ok=True)
        return anno_dir


def get_environment_config() -> Dict[str, Any]:
    """
    Get complete environment configuration.
    
    Returns:
        Dict[str, Any]: Environment configuration dictionary
    """
    config = {
        'is_colab': is_colab(),
        'is_jupyter': is_jupyter(),
        'data_root': get_data_root(),
        'image_root': get_image_root(),
        'weights_dir': get_weights_dir(),
        'analysis_dir': get_analysis_dir(),
        'anno_dir': get_anno_dir(),
        'train_img_dir': os.path.join(get_image_root(), 'train'),
        'test_img_dir': os.path.join(get_image_root(), 'test'),
    }
    return config


def setup_colab_environment():
    """
    Setup Google Colab environment (mount drive, etc.).
    """
    if is_colab():
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("Google Drive mounted successfully!")
        except Exception as e:
            print(f"Failed to mount Google Drive: {e}")
    else:
        print("Not running in Colab - no setup needed")


def print_environment_info():
    """
    Print environment information for debugging.
    """
    config = get_environment_config()
    
    print("=" * 60)
    print("ENVIRONMENT CONFIGURATION")
    print("=" * 60)
    print(f"Running in Colab: {config['is_colab']}")
    print(f"Running in Jupyter: {config['is_jupyter']}")
    print(f"Data root: {config['data_root']}")
    print(f"Image root: {config['image_root']}")
    print(f"Weights dir: {config['weights_dir']}")
    print(f"Analysis dir: {config['analysis_dir']}")
    print(f"Annotations dir: {config['anno_dir']}")
    print(f"Train images: {config['train_img_dir']}")
    print(f"Test images: {config['test_img_dir']}")
    print("=" * 60)