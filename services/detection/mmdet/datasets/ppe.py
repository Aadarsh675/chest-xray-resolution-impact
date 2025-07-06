# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets.coco import CocoDataset
from mmdet.registry import DATASETS

@DATASETS.register_module()
class PPEDataset(CocoDataset):
    """PPE dataset for detection."""
    
    METAINFO = {
        'classes': ('gloves', 
                   'hand', 
                   'head_glasses', 
                   'head_hardhat', 
                   'head_no_glasses', 
                   'head_no_hardhat',
                   'person', 
                   'person_high_vis', 
                   'person_no_high_vis', 
                   'person_harness', 
                   'person_sitting',
                   'person_on_break', 
                   'person_not_on_break', 
                   'person_lying_down'),
        'palette': [(220, 20, 60), 
                   (119, 11, 32), 
                   (0, 0, 142), 
                   (0, 0, 230),
                   (106, 0, 228), 
                   (0, 60, 100), 
                   (0, 80, 100), 
                   (0, 0, 70),
                   (0, 0, 192), 
                   (250, 170, 30), 
                   (100, 170, 30), 
                   (220, 220, 0),
                   (175, 116, 175), 
                   (250, 0, 30)]
    }