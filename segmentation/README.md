### Training
- python3 train.py --name DeepLabV3Plus_ResNet152_Adamp_0.0001_marginx_plus_plus_notb_pseudo_cos_kfold2 --model DeepLabV3Plus_ResNet152

- python3 train.py --name DeepLabV3Plus_ResNet152_Adamp_0.0001_marginx_plus_plus_notb_pseudo_cos_all --model DeepLabV3Plus_ResNet152

### Inference
- python3 inference.py --model_dir './model/DeepLabV3Plus_SeResNext101_Adamp_0.0001_cutout30_plus_pseudo_cos_softcrossentropyloss0.52' --output_name 'DeepLabV3Plus_SeResNext101_Adamp_0.0001_marginx_plus_plus_notb_pseudo_cos_crf7.csv' --model DeepLabV3Plus_SeResNext101

- python3 inference.py --model_dir './model/DeepLabV3Plus_ResNet152_Adamp_0.0001_marginx_plus_plus_notb_pseudo_cos_all' --output_name 'DeepLabV3Plus_ResNet152_Adamp_0.0001_marginx_plus_plus_notb_pseudo_cos_all.csv' --model DeepLabV3Plus_ResNet152