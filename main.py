"""
Main training and validation pipeline for chest X-ray resolution impact study.

This script handles the complete pipeline from data loading to model evaluation,
with environment-aware path configuration for both local and Google Colab execution.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.environment import get_environment_config, setup_colab_environment, print_environment_info
from src.train import main as train_main
from src.test import main as test_main


def main():
    """Main entry point for the training and validation pipeline."""
    
    # Environment setup
    # Setup environment and get configuration
    config = get_environment_config()
    
    # Setup Colab environment if needed
    if config['is_colab']:
        setup_colab_environment()
    
    # Print environment information
    print_environment_info()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Chest X-ray Resolution Impact Study')
    parser.add_argument('--mode', choices=['train', 'test', 'both'], default='both',
                       help='Mode to run: train, test, or both (default: both)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--model', type=str, default='rtmdet',
                       help='Model architecture to use (default: rtmdet)')
    parser.add_argument('--resolution', type=str, default='original',
                       help='Image resolution to use (default: original)')
    
    args = parser.parse_args()
    
    # Set up paths using environment configuration
    ANNO_DIR     = config['anno_dir']  # COCO jsons live here
    IMAGE_ROOT   = config['image_root']  # PNGs live here: train/ and test/
    WEIGHTS_DIR  = config['weights_dir']
    
    # Image directories
    TRAIN_IMG_DIR = config['train_img_dir']
    VAL_IMG_DIR   = config['train_img_dir']  # val JSON picks subset
    TEST_IMG_DIR  = config['test_img_dir']
    
    # Output directory for threshold curves
    out_dir = os.path.join(config['analysis_dir'], "scale_100", f"rep_{1}")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Annotations: {ANNO_DIR}")
    print(f"  Image root: {IMAGE_ROOT}")
    print(f"  Weights: {WEIGHTS_DIR}")
    print(f"  Train images: {TRAIN_IMG_DIR}")
    print(f"  Test images: {TEST_IMG_DIR}")
    print(f"  Output: {out_dir}")
    
    # Run training if requested
    if args.mode in ['train', 'both']:
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        # Set up training arguments
        train_args = argparse.Namespace(
            epochs=args.epochs,
            batch_size=args.batch_size,
            model=args.model,
            resolution=args.resolution,
            anno_dir=ANNO_DIR,
            image_root=IMAGE_ROOT,
            weights_dir=WEIGHTS_DIR,
            train_img_dir=TRAIN_IMG_DIR,
            val_img_dir=VAL_IMG_DIR,
            test_img_dir=TEST_IMG_DIR,
            output_dir=out_dir
        )
        
        try:
            train_main(train_args)
            print("Training completed successfully!")
        except Exception as e:
            print(f"Training failed: {e}")
            return 1
    
    # Run testing if requested
    if args.mode in ['test', 'both']:
        print("\n" + "="*60)
        print("STARTING TESTING")
        print("="*60)
        
        # Set up testing arguments
        test_args = argparse.Namespace(
            model=args.model,
            resolution=args.resolution,
            anno_dir=ANNO_DIR,
            image_root=IMAGE_ROOT,
            weights_dir=WEIGHTS_DIR,
            test_img_dir=TEST_IMG_DIR,
            output_dir=out_dir
        )
        
        try:
            test_main(test_args)
            print("Testing completed successfully!")
        except Exception as e:
            print(f"Testing failed: {e}")
            return 1
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED")
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
