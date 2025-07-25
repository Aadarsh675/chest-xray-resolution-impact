# Copyright (c) OpenMMLab. All rights reserved.

# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0

from mmengine.config import read_base

with read_base():
    from .._base_.default_runtime import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.datasets.ppe import *
    from .rtmdet_tta import *

from mmcv.ops import nms
from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms.processing import RandomResize
from mmengine.hooks.ema_hook import EMAHook
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.nn import SyncBatchNorm
from torch.nn.modules.activation import SiLU
from torch.optim.adamw import AdamW

from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms.transforms import (CachedMixUp, CachedMosaic,
                                                  Pad, RandomCrop, RandomFlip,
                                                  Resize, YOLOXHSVRandomAug)

from mmdet.engine.hooks.pipeline_switch_hook import PipelineSwitchHook
from mmdet.models.backbones.cspnext import CSPNeXt
from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.models.dense_heads.rtmdet_head import RTMDetSepBNHead
from mmdet.models.detectors.rtmdet import RTMDet
from mmdet.models.layers.ema import ExpMomentumEMA
from mmdet.models.losses.gfocal_loss import QualityFocalLoss
from mmdet.models.losses.iou_loss import GIoULoss
from mmdet.models.necks.cspnext_pafpn import CSPNeXtPAFPN
from mmdet.models.task_modules.assigners.dynamic_soft_label_assigner import \
    DynamicSoftLabelAssigner
from mmdet.models.task_modules.coders.distance_point_bbox_coder import \
    DistancePointBBoxCoder
from mmdet.models.task_modules.prior_generators.point_generator import \
    MlvlPointGenerator

num_classes=14
max_epochs = 300*5
stage2_num_epochs = 20*5
base_lr = 0.004
interval = 10

model = dict(
    type=RTMDet,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type=CSPNeXt,
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1,
        widen_factor=1,
        channel_attention=True,
        norm_cfg=dict(type=SyncBatchNorm),
        act_cfg=dict(type=SiLU, inplace=True)),
    neck=dict(
        type=CSPNeXtPAFPN,
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type=SyncBatchNorm),
        act_cfg=dict(type=SiLU, inplace=True)),
    bbox_head=dict(
        type=RTMDetSepBNHead,
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        anchor_generator=dict(
            type=MlvlPointGenerator, offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type=DistancePointBBoxCoder),
        loss_cls=dict(
            type=QualityFocalLoss, use_sigmoid=True, beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type=GIoULoss, loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type=SyncBatchNorm),
        act_cfg=dict(type=SiLU, inplace=True)),
    train_cfg=dict(
        assigner=dict(type=DynamicSoftLabelAssigner, topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type=nms, iou_threshold=0.65),
        max_per_img=300),
)

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(type=CachedMosaic, img_scale=(640, 640), pad_val=114.0),
    dict(
        type=RandomResize,
        scale=(1280, 1280),
        ratio_range=(0.1, 2.0),
        resize_type=Resize,
        keep_ratio=True),
    dict(type=RandomCrop, crop_size=(640, 640)),
    dict(type=YOLOXHSVRandomAug),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type=CachedMixUp,
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type=PackDetInputs)
]

train_pipeline_stage2 = [
    dict(type=LoadImageFromFile),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=RandomResize,
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        resize_type=Resize,
        keep_ratio=True),
    dict(type=RandomCrop, crop_size=(640, 640)),
    dict(type=YOLOXHSVRandomAug),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type=PackDetInputs)
]

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=Resize, scale=(640, 640), keep_ratio=True),
    dict(type=Pad, size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader.update(
    dict(
        batch_size=32,
        num_workers=10,
        batch_sampler=None,
        pin_memory=True,
        dataset=dict(pipeline=train_pipeline)))
val_dataloader.update(
    dict(batch_size=5, num_workers=10, dataset=dict(pipeline=test_pipeline)))
test_dataloader = val_dataloader

train_cfg.update(
    dict(
        max_epochs=max_epochs,
        val_interval=interval,
        dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)]))

val_evaluator.update(dict(proposal_nums=(100, 1, 10)))
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type=LinearLR, start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type=CosineAnnealingLR,
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks.update(
    dict(
        checkpoint=dict(
            interval=10, 
            max_keep_ckpts=3, 
            save_best='coco/bbox_mAP', 
            rule='greater',
            type='mmengine.hooks.CheckpointHook'),
            )
    )

custom_hooks = [
    dict(
        type=EMAHook,
        ema_type=ExpMomentumEMA,
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type=PipelineSwitchHook,
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]
