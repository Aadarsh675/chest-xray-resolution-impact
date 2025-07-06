#!/usr/bin/env python
#-*- coding: utf-8 -*-
#
# ==============================================================================
# Copyright (c) 2024 Nextera Robotic Systems
# Description: Utility functions for inference
#
# Author: Andrew Kent (modified)
# Created on: Thu Mar 07 2024
# ==============================================================================

import numpy as np
from PIL import Image
import cv2
import json

def load_class_colors(color_file):
    """
    Reads class_color.txt and constructs a dictionary: {class_name: (R, G, B)}
    
    Args:
        color_file: Path to the color file
        
    Returns:
        Dictionary mapping class names to RGB color tuples
    """
    color_dict = {}

    with open(color_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                # Skip blank lines or commented lines
                continue
            # Expecting lines like: "R G B class_name"
            parts = line.split(maxsplit=3)
            if len(parts) != 4:
                continue  # skip malformed lines
            r, g, b, class_name = parts
            color_dict[class_name] = (int(r), int(g), int(b))

    return color_dict

def get_color_for_class(class_name, class_colors):
    """
    Returns a color tuple (R, G, B) for a given class_name.
    
    Args:
        class_name: Name of the class
        class_colors: Dictionary mapping class names to colors
        
    Returns:
        RGB color tuple
    """
    if class_name in class_colors:
        return class_colors[class_name]
    # Default color: bright green
    return (0, 255, 0)

def load_json_file(file_path):
    """
    Load and parse a JSON file
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def resize_image(image_path, output_path, width, height, logger=None):
    """
    Resizes an image to the specified width and height.

    Args:
        image_path: Path to the input image
        output_path: Path to save the resized image
        width: Desired width of the resized image
        height: Desired height of the resized image
        logger: Logger instance (optional)
    """
    # Open the image file
    with Image.open(image_path) as img:
        
        # Ensure the image is in RGB mode (important for some formats)
        if img.mode != "RGB":
            img = img.convert("RGB")
            if logger:
                logger.info("Converted image to RGB mode.")

        # Resize the image
        resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
        
        # Save the resized image
        resized_img.save(output_path)
        
    return output_path

def compute_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: First bounding box in format [x, y, w, h] or [x1, y1, x2, y2]
        box2: Second bounding box in format [x, y, w, h] or [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    # For YOLOX format (x1, y1, w, h)
    if len(box1) == 4 and isinstance(box1[2], (int, float)) and box1[2] > 0:
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    # For format (x1, y1, x2, y2)
    else:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection area
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    iou = inter_area / (box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0
    return iou