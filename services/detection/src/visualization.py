#!/usr/bin/env python
#-*- coding: utf-8 -*-
#
# ==============================================================================
# Copyright (c) 2024 Nextera Robotic Systems
# Description: Visualization functions for inference results
#
# Author: Andrew Kent (modified)
# Created on: Thu Mar 07 2024
# ==============================================================================

import os
import cv2
import numpy as np

class Visualizer:
    """
    Handles visualization of detection results
    """

    def __init__(self, class_colors, model_type, logger):
        """
        Initialize the visualizer

        Args:
            class_colors: Dictionary mapping class names to colors
            model_type: Model type ('yolox' or 'solov2')
            logger: Logger instance
        """
        self.class_colors = class_colors
        self.model_type = model_type
        self.logger = logger

    def get_color_for_class(self, class_name):
        """
        Get color for a class

        Args:
            class_name: Class name

        Returns:
            RGB color tuple
        """
        if class_name in self.class_colors:
            return self.class_colors[class_name]
        # Default color: bright green
        return (0, 255, 0)

    def overlay_detections(self, image_path, detections):
        """
        Draw bounding boxes and segmentation masks on an image

        Args:
            image_path: Path to the input image
            detections: List of detection dictionaries

        Returns:
            Path to the output image
        """
        # Construct the output path in the detection folder
        output_path = image_path.replace("images", "detection")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")

        img_height, img_width = image.shape[:2]

        # Create a copy for segmentation overlays (needed for SOLOv2)
        overlay = image.copy()

        # First, draw segmentation masks if available (for SOLOv2)
        alpha = 0.35  # Transparency factor for segmentation masks

        if self.model_type == 'solov2' or 'ins' in self.model_type:
            for det in detections:
                if 'segmentation' in det:
                    # Get segmentation polygon
                    class_name = det['category_name']
                    color = self.get_color_for_class(class_name)

                    # Convert segmentation to a mask
                    try:
                        # If segmentation is in RLE format, process accordingly
                        if isinstance(det['segmentation'], dict):
                            # TODO: Implement RLE decoding if needed
                            pass
                        # If segmentation is polygon format
                        elif isinstance(det['segmentation'], list):
                            # Create a mask from polygon
                            mask = np.zeros((img_height, img_width), dtype=np.uint8)

                            # If it's a list of polygons (multiple contours)
                            for polygon in det['segmentation']:
                                contour = np.array(polygon).reshape(-1, 2).astype(np.int32)
                                cv2.fillPoly(mask, [contour], 1)

                                # Draw polygon outline
                                cv2.polylines(image, [contour], True, color, 2)

                            # Apply colored mask
                            colored_mask = np.zeros_like(image)
                            colored_mask[mask == 1] = color

                            # Blend the colored mask with the original image
                            cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0, overlay)
                    except Exception as e:
                        self.logger.warning(f"Error drawing segmentation: {e}")

        # Apply overlay to original image
        if self.model_type == 'solov2' or 'ins' in self.model_type:
            image = overlay

        # Now draw bounding boxes and labels
        for det in detections:
            x, y, w, h = det['bbox']
            class_name = det['category_name']
            color = self.get_color_for_class(class_name)
            confidence = det['score']

            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # Create label with class name and confidence
            label_text = f"{class_name} {confidence:.2f}"

            # Get size of text for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            # Draw background rectangle for text
            cv2.rectangle(
                image,
                (x, y - text_height - 5),
                (x + text_width, y),
                color,
                -1  # Fill rectangle
            )

            # Draw text (white)
            cv2.putText(
                image,
                label_text,
                (x, max(y - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text for better visibility
                2
            )

        # Save the image with detections to the detection folder
        cv2.imwrite(output_path, image)

        return output_path
