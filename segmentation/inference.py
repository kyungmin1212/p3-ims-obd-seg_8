import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from importlib import import_module
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision
import torchvision.transforms as transforms

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, unary_from_softmax

def inference(data_dir,model_dir,output_dir,args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.model == "DeepLabV3_ResNet50":
        model=smp.DeepLabV3(encoder_name='resnet50',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == 'DeepLabV3_ResNet101':
        model=smp.DeepLabV3(encoder_name='resnet101',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == 'DeepLabV3Plus_ResNet50':
        model=smp.DeepLabV3Plus(encoder_name='resnet50',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == 'DeepLabV3Plus_ResNet101':
        model=smp.DeepLabV3Plus(encoder_name='resnet101',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == 'DeepLabV3Plus_ResNet152':
        model=smp.DeepLabV3Plus(encoder_name='resnet152',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model =='DeepLabV3Plus_SeResNext101':
        model=smp.DeepLabV3Plus(encoder_name='se_resnext101_32x4d',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == 'PSPNet_ResNet50':
        model=smp.PSPNet(encoder_name='resnet50',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == 'PSPNet_ResNet101':
        model=smp.PSPNet(encoder_name='resnet101',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == 'Unet_ResNet50':
        model=smp.Unet(encoder_name='resnet50',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == 'Unet_ResNet101':
        model=smp.Unet(encoder_name='resnet101',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == 'UnetPlusPlus_ResNet50':
        model=smp.UnetPlusPlus(encoder_name='resnet50',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == 'UnetPlusPlus_ResNet101':
        model=smp.UnetPlusPlus(encoder_name='resnet101',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == "DeepLabV3_EfficientNetb0":
        model=smp.DeepLabV3(encoder_name='efficientnet-b0',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == 'DeepLabV3_EfficientNetb4':
        model=smp.DeepLabV3(encoder_name='efficientnet-b4',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == 'DeepLabV3Plus_EfficientNetb0':
        model=smp.DeepLabV3Plus(encoder_name='efficientnet-b0',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == 'DeepLabV3Plus_EfficientNetb4':
        model=smp.DeepLabV3Plus(encoder_name='efficientnet-b4',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == 'PSPNet_EfficientNetb0':
        model=smp.PSPNet(encoder_name='efficientnet-b0',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == 'PSPNet_EfficientNetb4':
        model=smp.PSPNet(encoder_name='efficientnet-b4',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == 'Unet_EfficientNetb0':
        model=smp.Unet(encoder_name='efficientnet-b0',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == 'Unet_EfficientNetb4':
        model=smp.Unet(encoder_name='efficientnet-b4',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == 'UnetPlusPlus_EfficientNetb0':
        model=smp.UnetPlusPlus(encoder_name='efficientnet-b0',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    elif args.model == 'UnetPlusPlus_EfficientNetb4':
        model=smp.UnetPlusPlus(encoder_name='efficientnet-b4',classes=args.num_classes,encoder_weights='imagenet',activation=None)
    else:
        assert False ,"모델 설정을 다시 하세요."
    
    model_path = os.path.join(model_dir, 'best.pth')
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.to(device)


    test_path=os.path.join(data_dir,args.test_name)
    dataset_module=getattr(import_module('dataset'),args.dataset)   # default : CustomDataLoader
    test_transform_module = getattr(import_module("dataset"), args.test_augmentation)  # default: TestAugmentation
    test_transform=test_transform_module()  # augmentation 에 입력값을 넣어주고 싶으면 TestAugmentation을 수정하고 여기에 넣어주면된다.
    # test dataset
    test_dataset = dataset_module(data_dir=test_path, mode='test', transform=test_transform)
    
    def collate_fn(batch):
        return tuple(zip(*batch))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            collate_fn=collate_fn)
    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)
    
    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    print('Start prediction.')
    model.eval()

    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)

    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(test_loader):

            invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                                 std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                            transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                                 std = [ 1., 1., 1. ]),
                                           ])

            # (batch,channel,512,512)
            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))

            for i in range(len(imgs)):

                image1=imgs[i]
                mask1=outs[i]
                image1=invTrans(image1)
                image1=image1.detach().cpu().numpy()
                mask1=mask1.detach().cpu().numpy()

                image1=image1.transpose(1,2,0)
                n_classes=12
                feat_first = mask1.reshape((n_classes,-1))
                unary = unary_from_softmax(feat_first)
                unary=np.ascontiguousarray(unary)
                d = dcrf.DenseCRF2D(image1.shape[1], image1.shape[0], n_classes)
                d.setUnaryEnergy(unary)
                d.addPairwiseGaussian(sxy=(5, 5), compat=3, kernel=dcrf.DIAG_KERNEL,
                                        normalization=dcrf.NORMALIZE_SYMMETRIC)

                d.addPairwiseBilateral(sxy=(50, 50), srgb=(13, 13, 13), rgbim=image1.astype(np.uint8),
                                    compat=10,
                                    kernel=dcrf.DIAG_KERNEL,
                                    normalization=dcrf.NORMALIZE_SYMMETRIC)

                Q = d.inference(5)
                res = np.argmax(Q, axis=0).reshape((image1.shape[0], image1.shape[1]))
                transformed = transform(image=image1, mask=res)
                mask=transformed['mask']
                mask=mask.reshape([1,256*256]).astype(int)
                preds_array=np.vstack((preds_array,mask))

            # oms = torch.argmax(outs, dim=1).detach().cpu().numpy()
            
            # # resize (256 x 256)
            # temp_mask = []
            # for img, mask in zip(np.stack(imgs), oms):
            #     transformed = transform(image=img, mask=mask)
            #     # (256,256)
            #     mask = transformed['mask']
            #     temp_mask.append(mask)
            # #(batch,256,256)
            # oms = np.array(temp_mask)
            # #(batch,256*256)
            # oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            # preds_array = np.vstack((preds_array, oms))

            # 배치가 쌓이다가 (배치+배치+배치+...,256*256) 
            # (전체데이터개수,256*256) 와 같이 된다.

            # i는 batch 중에서 한개 그 한개의 파일이름을 배치만큼모아놓은리스트[f1,f2,f3,..fbatch]를 file_name_list에 넣는다.
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    # file_names 를 리스트 에 있는거를 다 꺼내서 하나의 리스트에 이름을 그냥 쭉넣어준다.
    file_names = [y for x in file_name_list for y in x]

    for file_name, string in zip(file_names, preds_array):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(os.path.join(output_dir,f"{args.output_name}"), index=False)
    print(f'Inference Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=12)
    parser.add_argument('--model', type=str, default='DeepLabV3Plus_EfficientNetb4', help='model type (default: DeepLabV3Plus_EfficientNetb0)')
    parser.add_argument('--dataset',type=str,default='CustomDataLoader', help='dataset augmentation type (default: CustomDataLoader)')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for validing (default: 16)')
    parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data')
    parser.add_argument('--test_name',type=str, default='test.json')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--test_augmentation',type=str, default='TestAugmentation', help='data augmentation type (default: TestAugmentation)')
    parser.add_argument('--num_workers',type=int, default=2)
    # 반드시 설정
    # model 저장된 best.pth 가 있는 폴더명 지정해주기
    parser.add_argument('--model_dir', type=str, default='./model/deeplabv3plus_efficientnetb4') 
    # csv 파일 이름 지정해주기
    parser.add_argument('--output_name', type=str, default='output.csv') 

    args = parser.parse_args()

    
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)


    data_dir = args.data_dir
    model_dir = args.model_dir
    inference(data_dir,model_dir,output_dir,args)
