from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmcv.runner import load_checkpoint
classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
# config file 들고오기

# 수정
cfg = Config.fromfile('/opt/ml/realcode/swinb_htc_plus/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py')

PREFIX = '/opt/ml/input/data/'


# dataset 바꾸기
cfg.data.train.classes = classes
cfg.data.train.img_prefix = PREFIX
cfg.data.train.ann_file = PREFIX + 'train.json'
cfg.data.train.seg_prefix=PREFIX
# cfg.data.train.pipeline[2]['img_scale'] = (512, 512)

cfg.data.val.classes = classes
cfg.data.val.img_prefix = PREFIX
cfg.data.val.ann_file = PREFIX + 'val.json'

# cfg.data.val.pipeline[1]['img_scale'] = (512, 512)

cfg.data.test.classes = classes
cfg.data.test.img_prefix = PREFIX
cfg.data.test.ann_file = PREFIX + 'test.json'
# cfg.data.test.pipeline[1]['img_scale'] = (512, 512)

# cfg.data.samples_per_gpu = 4

cfg.seed=2020
cfg.gpu_ids = [0]

# 수정
cfg.work_dir = '/opt/ml/realcode/work_dirs/swinb_htc_plus'
# cfg.model.roi_head.bbox_head.num_classes = 11

# cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)

model = build_detector(cfg.model)
checkpoint_path = '/opt/ml/realcode/swinb_htc_plus/cascade_mask_rcnn_swin_base_patch4_window7.pth'
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cuda')

datasets = [build_dataset(cfg.data.train)]

train_detector(model, datasets[0], cfg, distributed=False, validate=True)