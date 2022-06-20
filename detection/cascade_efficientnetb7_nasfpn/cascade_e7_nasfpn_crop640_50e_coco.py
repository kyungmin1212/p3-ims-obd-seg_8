_base_ = [
    '/opt/ml/realcode/cascade_efficientnetb7_nasfpn/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco.py',
    # '/opt/ml/code/mmdetection_trash/configs/trash/dataset.py',
    # '/opt/ml/code/mmdetection_trash/configs/_base_/schedules/schedule_1x.py',
    # '/opt/ml/code/mmdetection_trash/configs/_base_/default_runtime.py'
]
cudnn_benchmark = True
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='CascadeRCNN',
    pretrained=True,
    backbone=dict(
        type='EfficientNet',
        model_type='efficientnet-b7',  # Possible types: ['efficientnet-b0' ... 'efficientnet-b7']
        out_indices=(0, 1, 3, 6)),  # Possible indices: [0 1 2 3 4 5 6],
    neck=dict(type='NASFPN', stack_times=7, norm_cfg=norm_cfg),
 
    # training and testing settings
    train_cfg=dict(assigner=dict(neg_iou_thr=0.5)))
# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='InstaBoost',
        action_candidate=('normal', 'horizontal', 'skip'),
        action_prob=(1, 0, 0),
        scale=(0.8, 1.2),
        dx=15,
        dy=15,
        theta=(-1, 1),
        color_prob=0.5,
        hflag=False,
        aug_ratio=0.5),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad',size=(512, 512)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad',size=(512, 512)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(norm_decay_mult=0, bypass_duplicate=True))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    step=[30, 40])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=25)

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

checkpoint_config = dict(max_keep_ckpts=5, interval=1)
