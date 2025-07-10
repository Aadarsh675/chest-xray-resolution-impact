"""
Model trainer for NIH Chest X-rays using YOLO and other models
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


class ModelTrainer:
    def __init__(self, data_dir, model_name='yolov8n', epochs=10, batch_size=32):
        """
        Initialize model trainer
        
        Args:
            data_dir: Directory containing COCO format dataset
            model_name: Model to use (yolov8n, yolov8s, etc.)
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.data_dir = data_dir
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Try to import ultralytics
        try:
            from ultralytics import YOLO
            self.framework = 'ultralytics'
        except ImportError:
            print("Installing ultralytics...")
            import subprocess
            subprocess.run(["pip", "install", "ultralytics"], check=True)
            from ultralytics import YOLO
            self.framework = 'ultralytics'
        
        # Initialize model
        self.model = None
        self.results = {}
        
    def train(self):
        """Train the model"""
        print(f"\nTraining {self.model_name} on {self.device}...")
        
        if self.framework == 'ultralytics':
            return self._train_ultralytics()
        else:
            return self._train_detectron2()
    
    def _train_ultralytics(self):
        """Train using Ultralytics YOLO"""
        from ultralytics import YOLO
        
        # Load model
        if self.model_name.startswith('yolov8'):
            # Use pretrained model
            self.model = YOLO(f'{self.model_name}.pt')
        else:
            # Custom model
            self.model = YOLO(self.model_name)
        
        # Data yaml path
        data_yaml = os.path.join(self.data_dir, 'data.yaml')
        
        # Train model
        results = self.model.train(
            data=data_yaml,
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=640,
            device=self.device,
            project=os.path.join(self.data_dir, 'runs'),
            name=f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            exist_ok=True,
            patience=10,
            save=True,
            plots=True,
            val=True
        )
        
        # Save results
        self.results = {
            'framework': 'ultralytics',
            'model': self.model_name,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'device': str(self.device),
            'metrics': {
                'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
                'recall': float(results.results_dict.get('metrics/recall(B)', 0))
            }
        }
        
        return self.results
    
    def _train_detectron2(self):
        """Train using Detectron2 (alternative)"""
        try:
            import detectron2
        except ImportError:
            print("Detectron2 not available, using YOLO instead")
            return self._train_ultralytics()
        
        from detectron2 import model_zoo
        from detectron2.engine import DefaultTrainer
        from detectron2.config import get_cfg
        from detectron2.data import DatasetCatalog, MetadataCatalog
        
        # Register dataset
        self._register_detectron2_dataset()
        
        # Configure model
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("nih_chest_train",)
        cfg.DATASETS.TEST = ("nih_chest_val",)
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = self.batch_size
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = self.epochs * 1000  # Approximate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15  # NIH diseases
        cfg.OUTPUT_DIR = os.path.join(self.data_dir, "detectron2_output")
        
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        # Train
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        # Get metrics
        from detectron2.evaluation import COCOEvaluator
        evaluator = COCOEvaluator("nih_chest_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
        val_loader = trainer.build_test_loader(cfg, "nih_chest_val")
        metrics = trainer.test(cfg, trainer.model, evaluators=[evaluator])
        
        self.results = {
            'framework': 'detectron2',
            'model': 'faster_rcnn_R_50_FPN',
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'device': str(self.device),
            'metrics': metrics
        }
        
        return self.results
    
    def test(self):
        """Test the trained model"""
        print("\nTesting model...")
        
        if self.framework == 'ultralytics':
            return self._test_ultralytics()
        else:
            return self._test_detectron2()
    
    def _test_ultralytics(self):
        """Test using Ultralytics YOLO"""
        if self.model is None:
            # Load best model
            model_path = self._find_best_model()
            if model_path:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
            else:
                print("No trained model found!")
                return {}
        
        # Test on test set
        test_annotations = os.path.join(self.data_dir, 'annotations', 'instances_test.json')
        
        # Load test data
        with open(test_annotations, 'r') as f:
            test_data = json.load(f)
        
        # Run inference on test images
        results = []
        images_dir = os.path.join(self.data_dir, 'images')
        
        print(f"Running inference on {len(test_data['images'])} test images...")
        
        for img_info in test_data['images'][:100]:  # Limit to 100 images for speed
            img_path = os.path.join(images_dir, img_info['file_name'])
            if os.path.exists(img_path):
                # Run inference
                pred_results = self.model(img_path, conf=0.25)
                
                # Process results
                for r in pred_results:
                    if r.boxes is not None:
                        results.append({
                            'image_id': img_info['id'],
                            'file_name': img_info['file_name'],
                            'predictions': {
                                'boxes': r.boxes.xyxy.cpu().numpy().tolist() if r.boxes.xyxy is not None else [],
                                'scores': r.boxes.conf.cpu().numpy().tolist() if r.boxes.conf is not None else [],
                                'classes': r.boxes.cls.cpu().numpy().tolist() if r.boxes.cls is not None else []
                            }
                        })
        
        # Calculate metrics
        test_results = {
            'total_images': len(test_data['images']),
            'images_processed': len(results),
            'predictions': results[:10],  # Save only first 10 for brevity
            'model_path': str(self.model.model_path if hasattr(self.model, 'model_path') else 'unknown')
        }
        
        # Create visualizations
        self._create_test_visualizations(results[:5], images_dir)
        
        return test_results
    
    def _test_detectron2(self):
        """Test using Detectron2"""
        # Similar to ultralytics but using detectron2
        pass
    
    def _find_best_model(self):
        """Find the best trained model"""
        runs_dir = os.path.join(self.data_dir, 'runs')
        if not os.path.exists(runs_dir):
            return None
        
        # Find latest training run
        train_dirs = [d for d in os.listdir(runs_dir) if d.startswith('train_')]
        if not train_dirs:
            return None
        
        latest_dir = sorted(train_dirs)[-1]
        weights_dir = os.path.join(runs_dir, latest_dir, 'weights')
        
        # Look for best.pt
        best_path = os.path.join(weights_dir, 'best.pt')
        if os.path.exists(best_path):
            return best_path
        
        # Look for last.pt
        last_path = os.path.join(weights_dir, 'last.pt')
        if os.path.exists(last_path):
            return last_path
        
        return None
    
    def _create_test_visualizations(self, results, images_dir):
        """Create visualizations of test results"""
        vis_dir = os.path.join(self.data_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        print("Creating test visualizations...")
        
        # Load categories
        with open(os.path.join(self.data_dir, 'annotations', 'instances_all.json'), 'r') as f:
            coco_data = json.load(f)
        
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Create visualizations for first few results
        for i, result in enumerate(results[:5]):
            img_path = os.path.join(images_dir, result['file_name'])
            if not os.path.exists(img_path):
                continue
            
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(img_array)
            ax.set_title(f"Predictions: {result['file_name']}")
            
            # Draw predictions
            if result['predictions']['boxes']:
                for box, score, cls in zip(
                    result['predictions']['boxes'],
                    result['predictions']['scores'],
                    result['predictions']['classes']
                ):
                    x1, y1, x2, y2 = box
                    cls_name = categories.get(int(cls) + 1, f'Class {int(cls)}')
                    
                    # Draw box
                    rect = plt.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        fill=False, color='red', linewidth=2
                    )
                    ax.add_patch(rect)
                    
                    # Add label
                    ax.text(
                        x1, y1 - 5,
                        f'{cls_name}: {score:.2f}',
                        color='red', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7)
                    )
            
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'test_prediction_{i+1}.png'), dpi=150)
            plt.close()
        
        print(f"✓ Saved visualizations to {vis_dir}")
    
    def _register_detectron2_dataset(self):
        """Register dataset for Detectron2"""
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.structures import BoxMode
        
        def get_nih_dicts(split='train'):
            """Load dataset in Detectron2 format"""
            json_file = os.path.join(self.data_dir, 'annotations', f'instances_{split}.json')
            with open(json_file) as f:
                data = json.load(f)
            
            # Create image id to annotations mapping
            img_anns = {}
            for ann in data['annotations']:
                img_id = ann['image_id']
                if img_id not in img_anns:
                    img_anns[img_id] = []
                img_anns[img_id].append(ann)
            
            # Convert to Detectron2 format
            dataset_dicts = []
            for img in data['images']:
                record = {}
                record["file_name"] = os.path.join(self.data_dir, 'images', img['file_name'])
                record["image_id"] = img['id']
                record["height"] = img['height']
                record["width"] = img['width']
                
                annos = img_anns.get(img['id'], [])
                objs = []
                for ann in annos:
                    obj = {
                        "bbox": ann['bbox'],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": ann['category_id'] - 1,  # 0-indexed
                        "iscrowd": ann['iscrowd']
                    }
                    objs.append(obj)
                record["annotations"] = objs
                dataset_dicts.append(record)
            
            return dataset_dicts
        
        # Register dataset
        for split in ['train', 'val', 'test']:
            DatasetCatalog.register(f"nih_chest_{split}", lambda s=split: get_nih_dicts(s))
            
        # Set metadata
        MetadataCatalog.get("nih_chest_train").set(thing_classes=DISEASE_CLASSES[:-1])  # Exclude 'No Finding'