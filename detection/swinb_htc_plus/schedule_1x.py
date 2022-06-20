# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])

lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    periods=[18,10,7,6,5,4],
    restart_weights = [1, 0.25, 0.2,0.15,0.1,0.1],
    min_lr_ratio=1e-6)
runner = dict(type='EpochBasedRunnerAmp', max_epochs=52)
