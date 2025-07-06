# Copyright (c) OpenMMLab. All rights reserved.

# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0

model_version = 'rtmdet_m_ppe_v0.1'
load_from = 'pretrained/rtmdet_m_8xb32-300e_coco.pth'  # noqa: E501

from mmengine.config import read_base
from mmengine.visualization import LocalVisBackend
from mmengine.visualization import WandbVisBackend

with read_base():
    from .rtmdet_l_8xb32_300e_coco import *

model.update(
    dict(
        backbone=dict(deepen_factor=0.67, widen_factor=0.75),
        neck=dict(
            in_channels=[192, 384, 768], out_channels=192, num_csp_blocks=2),
        bbox_head=dict(in_channels=192, feat_channels=192)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs={
            'project': 'Light_Weight_Tracking_V1',
            'entity': 'nexterarobotics',
            'name': model_version,
            'config': {
                'model_type': model_version,
                'batch_size': train_dataloader.get('batch_size', 32)
            }
        },
        define_metric_cfg={
            'loss': 'min',
            'loss_cls': 'min', 
            'loss_bbox': 'min',
            'coco/bbox_mAP': 'max',
            'coco/bbox_mAP_50': 'max',
            'coco/bbox_mAP_75': 'max',
            'learning_rate': None,
        },
        commit=True
    )
]

visualizer = dict(
    type=DetLocalVisualizer, vis_backends=vis_backends, name='visualizer')