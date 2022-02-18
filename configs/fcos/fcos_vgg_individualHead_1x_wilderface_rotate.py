_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='FCOS',
    backbone=dict(
        type='FCOS_VGG',
        out_indices=(2, 3, 4),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        norm_eval=True),
    neck=dict(
        type='FcosNeckKneron',
        in_channels=[128,196, 128],
        mid_channels=[128, 128, 64],
        out_channels=[256, 256, 128],
        out_kernel=1,
        out_padding=0,
        start_level=0,
        add_extra_convs='on_output',  # use P5
        num_outs=3,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOS_IndividualHead',
        num_classes=1,
        in_channels=[256, 256, 128], #, 128], #256,
        cls_kernel=1,
        reg_kernel=1,
        centerness_on_reg=True,
        stacked_convs=1,
        feat_channels=[256, 256, 256],
        strides=[8, 16, 32], #, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
img_norm_cfg = dict(
    mean=[128.0, 128.0, 128.0], std=[256.0, 256.0, 256.0], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Rotate90', angle=270),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Rotate90', angle=270),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


# optimizer
optimizer = dict(
    lr=0.001, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=144)

# Modify dataset related settings
dataset_type = 'CustomDataset'
classes = ('face',)
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=2,
    train=dict(
        pipeline=train_pipeline,
        type = 'CustomDataset',
        img_prefix='data/widerface/WIDER_train/images/',
        classes=classes,
        ann_file='data/widerface/retinaface_gt_v1.1/train/label.pkl'),
    val=dict(
        pipeline=test_pipeline,
        type = 'CustomDataset',
        img_prefix='data/widerface/WIDER_val/images/',
        classes=classes,
        ann_file='data/widerface/retinaface_gt_v1.1/val/label.pkl'),
    test=dict(
        pipeline=test_pipeline,
        type = 'CustomDataset',
        img_prefix='data/widerface/WIDER_val/images/',
        classes=classes,
        ann_file='data/widerface/retinaface_gt_v1.1/val/label.pkl'),)

# Change the evaluation metric since we use customized dataset.
# We can set the evaluation interval to reduce the evaluation times
evaluation = dict(interval=2, metric='mAP')
# We can set the checkpoint saving interval to reduce the storage cost
checkpoint_config = dict(interval=2)
# We can use the pre-trained Mask RCNN model to obtain higher performance
#load_from = 'checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
