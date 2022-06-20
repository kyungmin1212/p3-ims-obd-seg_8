_base_ = '/opt/ml/realcode/detectors_resnext101/htc_without_semantic_r50_fpn_1x_coco.py'
# _base_ = [
#     './dataset.py',
#     './schedule_1x.py', './default_runtime.py'
# ]


model = dict(
    backbone=dict(
        type='DetectoRS_ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True,
        style='pytorch',),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNeXt',
            depth=101,
            groups=32,
            base_width=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            pretrained='open-mmlab://resnext101_32x4d',
            style='pytorch')))

data = dict(samples_per_gpu=4, workers_per_gpu=2)
# learning policy
lr_config = dict(step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=14)
checkpoint_config = dict(max_keep_ckpts=3, interval=1)