import argparse
from loss import label_accuracy_score
import os
import glob
import re
import random
import json
import numpy as np
from importlib import import_module
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR,CosineAnnealingWarmRestarts

# pip install tensorboard
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
from adamp import AdamP
from adamp import SGDP
# $ pip install git+https://github.com/ildoonet/cutmix
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# 파일 이름이 중복될경우 뒤에 번호를 붙여주는 함수
def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

# train 진행중에 lr 확인용(lr_scheduler 를 사용하면 lr 이 바뀌기 때문)
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def rand_bbox(size, lamb):
    """ Generate random bounding box 
    Args:
        - size: [width, breadth] of the bounding box
        - lamb: (lambda) cut ratio parameter
    Returns:
        - Bounding box
    """
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lamb)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

#batch_image = (batch,channel,height,width)
def generate_cutmix_image(batch_image,batch_mask,beta=1):
    lam=np.random.beta(beta,beta)
    rand_index=np.random.permutation(len(batch_image)) # 배치수에서 임의로 순서로 순열을 만든다.

    bbx1, bby1, bbx2, bby2 = rand_bbox(batch_image[0].shape, lam)
    batch_image_updated = batch_image.clone()
    batch_mask_updated=batch_mask.clone()
    # 랜덤한 순열 배치대로 bbx1 bbx2 bby1 bby2 영역의 이미지를 바꿔준다.
    batch_image_updated[:, :,bbx1:bbx2, bby1:bby2] = batch_image[rand_index,:, bbx1:bbx2, bby1:bby2]
    batch_mask_updated[:,bbx1:bbx2,bby1:bby2]=batch_mask[rand_index,bbx1:bbx2,bby1:bby2]
    return batch_image_updated,batch_mask_updated


def train(data_dir,model_dir,args):
    seed_everything(args.seed)

    save_dir=increment_path(os.path.join(model_dir,args.name))

    use_cuda=torch.cuda.is_available()
    device=torch.device("cuda" if use_cuda else "cpu")
    
    train_path=os.path.join(data_dir,args.train_name)
    val_path=os.path.join(data_dir,args.val_name)


    # dataset_module 을 dataset.py 에서 args.dataset 클래스를 불러온다.
    dataset_module=getattr(import_module('dataset'),args.dataset)   # default : CustomDataLoader
    
    # augmentation 을 dataset.py 에서 TrainAugmentation 과 TestAugmentation 을 각각 불러온다.
    # train과 val 은 mask 가 있다.
    train_transform_module = getattr(import_module("dataset"), args.train_augmentation)  # default: TrainAugmentation
    val_transform_module = getattr(import_module("dataset"), args.val_augmentation)
    # test 는 mask 가 없다.
 
    
    train_transform=train_transform_module()    # augmentation 에 입력값을 넣어주고 싶으면 TrainAugmentation을 수정하고 여기에 넣어주면된다.
    val_transform=val_transform_module()  


    # train dataset
    train_dataset = dataset_module(data_dir=train_path, mode='train', transform=train_transform,cutbbox=args.cutbbox,margin=args.margin)

    # validation dataset
    val_dataset = dataset_module(data_dir=val_path, mode='val', transform=val_transform,cutbbox=args.cutbbox,margin=args.margin)

    

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=args.num_workers,
                                           collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            collate_fn=collate_fn)


    
    #!pip install git+https://github.com/qubvel/segmentation_models.pytorch
    
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
    elif args.model =='DeepLabV3Plus_SeResNet152':
        model=smp.DeepLabV3Plus(encoder_name='se_resnet152',classes=args.num_classes,encoder_weights='imagenet',activation=None)
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

    model.to(device)
    # 직접 구성한 클래스를 가져올때는 아래와 같은 코드를 이용한다. 
    # model_module = getattr(import_module("model"), args.model)  # default: BaseModel

    from pytorch_toolbelt import losses as L

    # https://smp.readthedocs.io/en/latest/losses.html#softcrossentropyloss
    # https://github.com/BloodAxe/pytorch-toolbelt
    if args.criterion == "cross_entropy":
        criterion=nn.CrossEntropyLoss()
    elif args.criterion == 'cutmixcrossentropyloss':
        criterion=CutMixCrossEntropyLoss(True)
    elif args.criterion == 'softcrossentropyloss':
        criterion=L.SoftCrossEntropyLoss(smooth_factor=0.5)
    else:
        assert False ,"loss 설정을 다시 하세요."

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr, weight_decay=1e-6)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params = model.parameters(), lr = args.lr,momentum=0.9)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-6)
    # pip3 install adamp
    elif args.optimizer == 'AdamP':
        optimizer = AdamP(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-6)
    elif args.optimizer == 'SGDP':
        optimizer = SGDP(params=model.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9, nesterov=True)
    else:
        assert False ,"optimizer 설정을 다시 하세요."

    if args.lr_scheduler=='basic':
        scheduler_check=False
    elif args.lr_scheduler == "StepLR":
        scheduler_check=True
        scheduler=StepLR(optimizer, args.lr_decay_step, gamma=0.5)
         #gamma 비율로 lr 감소 lr_decay_step 몇 epoch 마다 lr를 감소시킬건가
    elif args.lr_scheduler=='CosineAnnealingWarmRestarts':
        scheduler_check=True
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=1, eta_min=args.min_lr)
    else:
        assert False ,"lr_scheduler 설정을 다시 하세요."
    # -- logging
    # save_dir 에 log 를 찍기위함 SummaryWriter 
    # config 값을 save_dir/config.json 에 저장하는것이다.
    logger =SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir,'config.json'),'w',encoding='utf-8') as f:
        json.dump(vars(args),f,ensure_ascii=False,indent=4)
        # ensure_ascii =False 는 아스키 값으로 변환하지 말고 원래의 인코드를 유지하라는것
        # indent = 4 는 들여쓰기의 스페이스
    
    print('Start training..')
    best_loss=9999999
    best_mIoU=0
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.epochs):
        model.train()
        for step,(images,masks,_) in enumerate(train_loader):
            images=torch.stack(images)
            masks=torch.stack(masks).long()

            if args.cutmix:
                images,masks=generate_cutmix_image(images,masks)

            images,masks=images.to(device),masks.to(device)


            optimizer.zero_grad()
            if args.amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss=criterion(outputs,masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss=criterion(outputs,masks)
                loss.backward()
                optimizer.step()
    
            if (step+1)% args.log_interval == 0:
                current_lr=get_lr(optimizer)
                print(f"Epoch [{epoch+1}/{args.epochs}],Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}, lr : {current_lr:.7f}")
        if epoch==6:
            torch.save(model.state_dict(), f"{save_dir}/best7.pth")
        elif epoch==7:
            torch.save(model.state_dict(), f"{save_dir}/best8.pth")
        elif epoch==8:
            torch.save(model.state_dict(), f"{save_dir}/best9.pth")
        elif epoch==9:
            torch.save(model.state_dict(), f"{save_dir}/best10.pth")
        torch.save(model.state_dict(), f"{save_dir}/best.pth")
        if scheduler_check:
            scheduler.step()
        
        # # 매 valid 마다 출력하기(1) , valid
        # if (epoch+1)%args.val_log_interval==0:
        #     print(f'Start validataion #{epoch+1}')
        #     model.eval()
        #     with torch.no_grad():
        #         total_loss=0
        #         cnt=0
        #         mIoU_list=[]
        #         for step,(images,masks,_) in enumerate(val_loader):

        #             images=torch.stack(images)      # (batch, channel, height, width)
        #             masks=torch.stack(masks).long() # (batch, height, width)

        #             images,masks=images.to(device),masks.to(device)

        #             outputs=model(images) # (batch, channel(분류개수), height, width)
        #             loss=criterion(outputs,masks) # 자동으로 crossentropyloss 가 label 을 원핫 인코딩으로 바꿔주고 output값과 비교를해서 loss 를 구해준다.
        #             total_loss+=loss
        #             cnt+=1

        #             # 분류 개수 만큼 채널중에서 제일 큰 인덱스만 그 픽셀에 번호가 남게 된다.(batch,height,width) 가 된다.
        #             # 이제 masks 랑 비교해서 label_accuracy_score의 결과중에 2번째 결과값인 mIoU를 가져온다.
        #             # label_accuracy_score는 acc, acc_cls, mean_iu, fwavacc 를 반환한다. 
        #             outputs=torch.argmax(outputs,dim=1).detach().cpu().numpy()
        #             mIoU=label_accuracy_score(masks.detach().cpu().numpy(),outputs,n_class=args.num_classes)[2]
        #             mIoU_list.append(mIoU)
        #         avrg_mIoU=np.mean(mIoU_list)
        #         avrg_loss=total_loss/cnt
        #         print(f'Validataion #{epoch+1}  Average Loss: {avrg_loss:.4f},  mIoU : {avrg_mIoU:.4f} ')
        #     if avrg_mIoU>best_mIoU:
        #         print('Best performance at epoch: {}'.format(epoch + 1))
        #         print('Save model in', save_dir)
        #         best_mIoU = avrg_mIoU
        #         torch.save(model.state_dict(), f"{save_dir}/best.pth")
        #     torch.save(model.state_dict(), f"{save_dir}/last.pth")
        #     logger.add_scalar('Val/loss',avrg_loss,epoch)
        #     logger.add_scalar('Val/mIoU',avrg_mIoU,epoch)




if __name__=='__main__':
    parser=argparse.ArgumentParser()

    # 고정
    parser.add_argument('--num_classes', type=int, default=12)
    parser.add_argument('--seed',type=int,default=21,help='random seed (default :21')
    parser.add_argument('--dataset',type=str,default='CustomDataLoader', help='dataset augmentation type (default: CustomDataLoader)')
    parser.add_argument('--log_interval', type=int, default=25, help='how many batches to wait before logging training status')
    parser.add_argument('--val_log_interval', type=int, default=1)
    parser.add_argument('--train_augmentation',type=str, default='TrainAugmentation', help='data augmentation type (default: TrainAugmentation)')
    parser.add_argument('--val_augmentation',type=str, default='ValAugmentation', help='data augmentation type (default: TrainAugmentation)')
    parser.add_argument('--amp',type=bool,default=True)
    parser.add_argument('--cutbbox',type=bool,default=False)
    parser.add_argument('--margin',type=int,default=10)
    parser.add_argument('--cutmix',type=bool,default=False)

    parser.add_argument('--epochs',type=int,default=80,help='number of epochs to train (default :20')
    parser.add_argument('--name', default='exp', help='model save at {model_dir}/{name}')
    parser.add_argument('--batch_size', type=int, default=24, help='input batch size for training (default: 16)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model', type=str, default='DeepLabV3Plus_ResNet101', help='model type (default: DeepLabV3Plus_EfficientNetb4)')
    parser.add_argument('--criterion', type=str, default='softcrossentropyloss', help='criterion type (default: cross_entropy)')
    parser.add_argument('--optimizer', type=str, default='AdamP', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 1e-3)')
    parser.add_argument('--lr_scheduler', default='CosineAnnealingWarmRestarts')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='(StepLR)')
    parser.add_argument('--T_0', type=int, default=10, help='(CosineAnnealingWarmRestarts)')
    parser.add_argument('--min_lr', type=int, default=1e-7, help='(CosineAnnealingWarmRestarts)')
  # 데이터 경로 ,json파일 이름,모델 저장위치
    parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data')
    parser.add_argument('--train_name',type=str, default='train_all.json')
    parser.add_argument('--val_name',type=str, default='val.json')
    parser.add_argument('--model_dir', type=str, default='./model')

    args=parser.parse_args()
    print(args)

    data_dir=args.data_dir
    model_dir=args.model_dir

    train(data_dir,model_dir,args)