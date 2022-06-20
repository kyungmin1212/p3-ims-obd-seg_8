
# 원래 models 의 기본은 faster_rcnn_r50_fpn 거에서 가져왔다. 그걸 기준으로 수정을 하나씩 해나가는것
_base_ = [
    '/opt/ml/code/mmdetection_trash/configs/_base_/models/cascade_rcnn_r50_fpn.py',
    '/opt/ml/code/mmdetection_trash/configs/trash/dataset.py',
    '/opt/ml/code/mmdetection_trash/configs/_base_/schedules/schedule_1x.py',
    '/opt/ml/code/mmdetection_trash/configs/_base_/default_runtime.py'
]

# cascade_rcnn_r50_fpn.py 에서 num_classes 를 우리 모델에 맞게 수정을 해줘야하기 때문에
# cascade_rcnn_r50_fpn.py 에 들어가서 roi_head 에서 bbox_head 부분만 가져와서 num_classes 를 수정시켜준다.

# resnet 50 에서 resnext101 로 바꾸기 위해서는 backbone 을 새롭게 가져와줘야한다.
# 수정해주기 위해서는 이미 cascade_rcnn_x101_32x4d_fpn_20e_coco.py 에서처럼 x101 이 적용된 모델을 참고해서 가져온다.
model = dict(
    pretrained='open-mmlab://resnext101_32x4d',
    pretrained=True,
    backbone=dict(
        type='EfficientNet',
        model_type='efficientnet-b7',  # Possible types: ['efficientnet-b0' ... 'efficientnet-b7']
        out_indices=(0, 1, 3, 6)),  # Possible indices: [0 1 2 3 4 5 6],
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]
    )
)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

checkpoint_config = dict(max_keep_ckpts=3, interval=1)
