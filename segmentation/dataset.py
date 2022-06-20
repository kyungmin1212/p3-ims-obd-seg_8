from albumentations.augmentations.transforms import HorizontalFlip, VerticalFlip
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import json

import numpy as np
import pandas as pd

from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

class TrainAugmentation:
    def __init__(self,):
        self.transform = A.Compose([
                            A.Cutout(num_holes=30, max_h_size=40, max_w_size=40, p=1.0),
                            A.Normalize(
                                mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
                            ), 
                            ToTensorV2(transpose_mask=True)
                            ])

    # init 에 있는 변수명을 객체를 만들때 사용하고 그 객체를 다시 불러서 사용할때는 __call__ 이 실행된다.
    # 객체를 만들고 그 객체명에다가 안에 입력값을 넣어주면된다. 자동으로 return 값이 반환
    # 예를 들면 a=TrainAugmentation(init에 입력해야할 변수)
    # images,masks=a(image=images,mask=masks) 이렇게 실행하면 되는것이다.   
    def __call__(self,image,mask):
        return self.transform(image=image,mask=mask)['image'],self.transform(image=image,mask=mask)['mask']

class ValAugmentation:
    def __init__(self,**args):
        self.transform = A.Compose([
                            A.Normalize(
                                mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
                            ), 
                            ToTensorV2(transpose_mask=True)
                            ])

    # init 에 있는 변수명을 객체를 만들때 사용하고 그 객체를 다시 불러서 사용할때는 __call__ 이 실행된다.
    # 객체를 만들고 그 객체명에다가 안에 입력값을 넣어주면된다. 자동으로 return 값이 반환
    # 예를 들면 a=TrainAugmentation(init에 입력해야할 변수)
    # images,masks=a(image=images,mask=masks) 이렇게 실행하면 되는것이다.   
    def __call__(self,image,mask):
        return self.transform(image=image,mask=mask)['image'],self.transform(image=image,mask=mask)['mask']


class TestAugmentation:
    def __init__(self,**args):
        self.transform = A.Compose([
                            
                            A.Normalize(
                                mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
                            ),                           
                            ToTensorV2(transpose_mask=True)
                            ])
                  
    def __call__(self,image):
        return self.transform(image=image)['image']

def getpluslist(data_dir):
    anns_file_path = data_dir

    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())
    plus_ids=[]
    coco = COCO(anns_file_path)
    images=dataset['images']
    for i in range(len(images)):
        image_id = coco.getImgIds(imgIds=i)
        image_infos = coco.loadImgs(image_id)[0]
        ann_ids =coco.getAnnIds(imgIds=image_infos['id'])
        anns =coco.loadAnns(ann_ids)
        #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        #[{'id': 0, 'image_id': 0, 'category_id': 8, 'segmentation': [[184, 455, 183, 455, 182,,...]],'bbox':[ , , , ]
        for j in range(len(anns)):
            if anns[j]['category_id']==9:
                plus_ids.append(anns[j]['image_id'])
            elif anns[j]['category_id']==10:
                plus_ids.append(anns[j]['image_id'])
            elif anns[j]['category_id']==4:
                plus_ids.append(anns[j]['image_id'])
            elif anns[j]['category_id']==5:
                plus_ids.append(anns[j]['image_id'])
            elif anns[j]['category_id']==3:
                plus_ids.append(anns[j]['image_id'])

    return list(set(plus_ids))

import pandas as pd

def pseudodata():
    # cs=pd.read_csv('./output/DeepLabV3Plus_ResNet152_Adamp_0.0001_marginx_plus_plus_notb_pseudo_cos.csv')
    cs=pd.read_csv('./output/DeepLabV3Plus_ResNet152_Adamp_0.0001_cutout_plus_pseudo_cos.csv')
    file_name=cs['image_id']
    data=cs['PredictionString']
    new_data=[]
    for i in range(len(file_name)):
        if len(data[i])==131071:
            check_data=' '.join(data[i]).split()
            check_data=np.array(check_data)
            check_data=check_data.reshape(256,256)
            new_data.append([file_name[i],check_data])
    new_data=np.array(new_data)
    return new_data


class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, data_dir, mode = 'train', transform = None,cutbbox=False,margin=50):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)   # json 파일을 가져와 annotation 에 쉽게 접근하게 해주는 라이브러리다.
        # category_names 을 정의를 해준것이다. ( 우리가 분류할 클래스 이름들)
        self.category_names=['Background', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
        self.cutbbox=cutbbox
        self.margin=margin
        self.data_dir=data_dir
        if (self.mode in ('train', 'val')):
            self.plus_list=getpluslist(self.data_dir)
        self.pseudodata=pseudodata()
        
    def get_classname(self,classID, cats):
        for i in range(len(cats)):
            if cats[i]['id']==classID:
                return cats[i]['name']
        return "None"

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        if self.mode!='test' and index>=2*(len(self.coco.getImgIds())+len(self.plus_list)+len(self.plus_list)+len(self.plus_list)):
            idx=index-2*(len(self.coco.getImgIds())+len(self.plus_list)+len(self.plus_list)+len(self.plus_list))
            filename=self.pseudodata[idx][0]
            masks=self.pseudodata[idx][1]
            masks = masks.astype(np.float32)
            dataset_path = '/opt/ml/input/data'
            images = cv2.imread(os.path.join(dataset_path, filename)) # dataset_path/file_name
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
            image_infos=1
            trfm=A.Compose([
                            A.Resize(512,512)
                            ])
            images,masks =trfm(image=images, mask=masks)['image'],trfm(image=images, mask=masks)['mask']
            if self.transform is not None:
                images,masks = self.transform(image=images, mask=masks)

            return images, masks, image_infos

        elif self.mode!='test'and index>=(len(self.coco.getImgIds())+len(self.plus_list)+len(self.plus_list)+len(self.plus_list)+len(self.coco.getImgIds())+len(self.plus_list)+len(self.plus_list)):
            idx=index-len(self.coco.getImgIds())-len(self.plus_list)-len(self.plus_list)-len(self.plus_list)-len(self.coco.getImgIds())-len(self.plus_list)-len(self.plus_list)
            image_infos = self.coco.loadImgs(self.plus_list[idx])[0]
            dataset_path = '/opt/ml/input/data'
            images = cv2.imread(os.path.join(dataset_path, image_infos['file_name'])) # dataset_path/file_name
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
            if (self.mode in ('train', 'val')):
                ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
                anns = self.coco.loadAnns(ann_ids)
                cat_ids = self.coco.getCatIds()
                cats = self.coco.loadCats(cat_ids)
                masks = np.zeros((image_infos["height"], image_infos["width"]))
                if self.cutbbox:
                    x1_min=512
                    y1_min=512
                    x1_max=0
                    y1_max=0
                for i in range(len(anns)):
                    if self.cutbbox:
                        xmin,ymin,width,height=anns[i]['bbox']
                        xmax=xmin+width
                        ymax=ymin+height
                        if x1_min>xmin:
                            x1_min=xmin
                        if y1_min>ymin:
                            y1_min=ymin
                        if x1_max<xmax:
                            x1_max=xmax
                        if y1_max<ymax:
                            y1_max=ymax
                    className = self.get_classname(anns[i]['category_id'], cats)
                    # 그 카테고리 이름에 해당하는 id 값을 가져온다 (단 여기서는 category_names(0 background가 추가된 리스트) 의 인덱스를 가져오는것이다)
                    pixel_value = self.category_names.index(className)
                    # annToMask 는 annotation값을 넣어주면 그 annotation 값에 해당하게 0과 1로 변경시켜준다.(여기서 여러개의 카테고리가 있을수 없다. 하나의 annotation 은 이미지 하나와 카테고리 id 하나만 지정하기 때문이다. 다만 객체는 여러개가 있을수가 있다. 그거를 0과 1로 만 표현해준것이다.)
                    masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
                masks = masks.astype(np.float32)
                if self.cutbbox and self.mode=="train":
                    if x1_min<self.margin:
                        x1_min=0
                    else:
                        x1_min-=self.margin
                    if y1_min<self.margin:
                        y1_min=0
                    else:
                        y1_min-=self.margin
                    if x1_max>(512-self.margin):
                        x1_max=512
                    else:
                        x1_max+=self.margin
                    if y1_max>(512-self.margin):
                        y1_max=512
                    else:
                        y1_max+=self.margin
                    x1_max=int(x1_max)
                    x1_min=int(x1_min)
                    y1_max=int(y1_max)
                    y1_min=int(y1_min)
                    if (x1_max-x1_min)>(y1_max-y1_min):
                        a=x1_max-x1_min
                        b=y1_max-y1_min
                        remain=a-b
                        while remain!=0:
                            if remain>=2:
                                if y1_min!=0 and y1_max!=512:
                                    y1_min-=1
                                    y1_max+=1
                                elif y1_min==0 and y1_max!=512:
                                    y1_max+=2
                                elif y1_min!=0 and y1_max==512:
                                    y1_min-=2
                                remain-=2
                            elif remain==1:
                                if y1_min!=0 and y1_max!=512:
                                    y1_min-=1
                                elif y1_min==0 and y1_max!=512:
                                    y1_max+=1
                                elif y1_min!=0 and y1_max==512:
                                    y1_min-=1
                                remain-=1
                    elif (x1_max-x1_min)<(y1_max-y1_min):
                        a=x1_max-x1_min
                        b=y1_max-y1_min
                        remain=b-a
                        while remain!=0:
                            if remain>=2:
                                if x1_min!=0 and x1_max!=512:
                                    x1_min-=1
                                    x1_max+=1
                                elif x1_min==0 and x1_max!=512:
                                    x1_max+=2
                                elif x1_min!=0 and x1_max==512:
                                    x1_min-=2
                                remain-=2
                            elif remain==1:
                                if x1_min!=0 and x1_max!=512:
                                    x1_min-=1
                                elif x1_min==0 and x1_max!=512:
                                    x1_max+=1
                                elif x1_min!=0 and x1_max==512:
                                    x1_min-=1
                                remain-=1

                    trfm=A.Compose([
                                    A.Crop(x_min=int(x1_min), y_min=int(y1_min), x_max=int(x1_max), y_max=int(y1_max),always_apply=False, p=1.0),
                                    A.Resize(512,512,p=1.0)
                                    ])
                    images,masks = trfm(image=images, mask=masks)['image'],trfm(image=images, mask=masks)['mask']
                trfm=A.Compose([
                                A.Rotate((45,45),p=1.0),
                                A.HorizontalFlip(p=1.0)
                                ])
                images,masks = trfm(image=images, mask=masks)['image'],trfm(image=images, mask=masks)['mask']
                if self.transform is not None:
                    images,masks = self.transform(image=images, mask=masks)
                
                return images, masks, image_infos

            if self.mode == 'test':
                # transform -> albumentations 라이브러리 활용
                if self.transform is not None:
                    images = self.transform(image=images)
    
                return images, image_infos

        elif self.mode!='test'and index>=(len(self.coco.getImgIds())+len(self.plus_list)+len(self.plus_list)+len(self.plus_list)+len(self.coco.getImgIds())+len(self.plus_list)):
            idx=index-len(self.coco.getImgIds())-len(self.plus_list)-len(self.plus_list)-len(self.plus_list)-len(self.coco.getImgIds())-len(self.plus_list)
            image_infos = self.coco.loadImgs(self.plus_list[idx])[0]
            dataset_path = '/opt/ml/input/data'
            images = cv2.imread(os.path.join(dataset_path, image_infos['file_name'])) # dataset_path/file_name
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
            if (self.mode in ('train', 'val')):
                ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
                anns = self.coco.loadAnns(ann_ids)
                cat_ids = self.coco.getCatIds()
                cats = self.coco.loadCats(cat_ids)
                masks = np.zeros((image_infos["height"], image_infos["width"]))
                if self.cutbbox:
                    x1_min=512
                    y1_min=512
                    x1_max=0
                    y1_max=0
                for i in range(len(anns)):
                    if self.cutbbox:
                        xmin,ymin,width,height=anns[i]['bbox']
                        xmax=xmin+width
                        ymax=ymin+height
                        if x1_min>xmin:
                            x1_min=xmin
                        if y1_min>ymin:
                            y1_min=ymin
                        if x1_max<xmax:
                            x1_max=xmax
                        if y1_max<ymax:
                            y1_max=ymax
                    className = self.get_classname(anns[i]['category_id'], cats)
                    # 그 카테고리 이름에 해당하는 id 값을 가져온다 (단 여기서는 category_names(0 background가 추가된 리스트) 의 인덱스를 가져오는것이다)
                    pixel_value = self.category_names.index(className)
                    # annToMask 는 annotation값을 넣어주면 그 annotation 값에 해당하게 0과 1로 변경시켜준다.(여기서 여러개의 카테고리가 있을수 없다. 하나의 annotation 은 이미지 하나와 카테고리 id 하나만 지정하기 때문이다. 다만 객체는 여러개가 있을수가 있다. 그거를 0과 1로 만 표현해준것이다.)
                    masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
                masks = masks.astype(np.float32)
                if self.cutbbox and self.mode=="train":
                    if x1_min<self.margin:
                        x1_min=0
                    else:
                        x1_min-=self.margin
                    if y1_min<self.margin:
                        y1_min=0
                    else:
                        y1_min-=self.margin
                    if x1_max>(512-self.margin):
                        x1_max=512
                    else:
                        x1_max+=self.margin
                    if y1_max>(512-self.margin):
                        y1_max=512
                    else:
                        y1_max+=self.margin
                    x1_max=int(x1_max)
                    x1_min=int(x1_min)
                    y1_max=int(y1_max)
                    y1_min=int(y1_min)
                    if (x1_max-x1_min)>(y1_max-y1_min):
                        a=x1_max-x1_min
                        b=y1_max-y1_min
                        remain=a-b
                        while remain!=0:
                            if remain>=2:
                                if y1_min!=0 and y1_max!=512:
                                    y1_min-=1
                                    y1_max+=1
                                elif y1_min==0 and y1_max!=512:
                                    y1_max+=2
                                elif y1_min!=0 and y1_max==512:
                                    y1_min-=2
                                remain-=2
                            elif remain==1:
                                if y1_min!=0 and y1_max!=512:
                                    y1_min-=1
                                elif y1_min==0 and y1_max!=512:
                                    y1_max+=1
                                elif y1_min!=0 and y1_max==512:
                                    y1_min-=1
                                remain-=1
                    elif (x1_max-x1_min)<(y1_max-y1_min):
                        a=x1_max-x1_min
                        b=y1_max-y1_min
                        remain=b-a
                        while remain!=0:
                            if remain>=2:
                                if x1_min!=0 and x1_max!=512:
                                    x1_min-=1
                                    x1_max+=1
                                elif x1_min==0 and x1_max!=512:
                                    x1_max+=2
                                elif x1_min!=0 and x1_max==512:
                                    x1_min-=2
                                remain-=2
                            elif remain==1:
                                if x1_min!=0 and x1_max!=512:
                                    x1_min-=1
                                elif x1_min==0 and x1_max!=512:
                                    x1_max+=1
                                elif x1_min!=0 and x1_max==512:
                                    x1_min-=1
                                remain-=1

                    trfm=A.Compose([
                                    A.Crop(x_min=int(x1_min), y_min=int(y1_min), x_max=int(x1_max), y_max=int(y1_max),always_apply=False, p=1.0),
                                    A.Resize(512,512,p=1.0)
                                    ])
                    images,masks = trfm(image=images, mask=masks)['image'],trfm(image=images, mask=masks)['mask']
                trfm=A.Compose([
                                A.Rotate((15,15),p=1.0),
                                A.HorizontalFlip(p=1.0)
                                ])
                images,masks = trfm(image=images, mask=masks)['image'],trfm(image=images, mask=masks)['mask']
                if self.transform is not None:
                    images,masks = self.transform(image=images, mask=masks)
                
                return images, masks, image_infos

            if self.mode == 'test':
                # transform -> albumentations 라이브러리 활용
                if self.transform is not None:
                    images = self.transform(image=images)
    
                return images, image_infos

        elif self.mode!='test'and index>=(len(self.coco.getImgIds())+len(self.plus_list)+len(self.plus_list)+len(self.plus_list)+len(self.coco.getImgIds())):
            idx=index-len(self.coco.getImgIds())-len(self.plus_list)-len(self.plus_list)-len(self.plus_list)-len(self.coco.getImgIds())
            image_infos = self.coco.loadImgs(self.plus_list[idx])[0]
            dataset_path = '/opt/ml/input/data'
            images = cv2.imread(os.path.join(dataset_path, image_infos['file_name'])) # dataset_path/file_name
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
            if (self.mode in ('train', 'val')):
                ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
                anns = self.coco.loadAnns(ann_ids)
                cat_ids = self.coco.getCatIds()
                cats = self.coco.loadCats(cat_ids)
                masks = np.zeros((image_infos["height"], image_infos["width"]))
                if self.cutbbox:
                    x1_min=512
                    y1_min=512
                    x1_max=0
                    y1_max=0
                for i in range(len(anns)):
                    if self.cutbbox:
                        xmin,ymin,width,height=anns[i]['bbox']
                        xmax=xmin+width
                        ymax=ymin+height
                        if x1_min>xmin:
                            x1_min=xmin
                        if y1_min>ymin:
                            y1_min=ymin
                        if x1_max<xmax:
                            x1_max=xmax
                        if y1_max<ymax:
                            y1_max=ymax
                    className = self.get_classname(anns[i]['category_id'], cats)
                    # 그 카테고리 이름에 해당하는 id 값을 가져온다 (단 여기서는 category_names(0 background가 추가된 리스트) 의 인덱스를 가져오는것이다)
                    pixel_value = self.category_names.index(className)
                    # annToMask 는 annotation값을 넣어주면 그 annotation 값에 해당하게 0과 1로 변경시켜준다.(여기서 여러개의 카테고리가 있을수 없다. 하나의 annotation 은 이미지 하나와 카테고리 id 하나만 지정하기 때문이다. 다만 객체는 여러개가 있을수가 있다. 그거를 0과 1로 만 표현해준것이다.)
                    masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
                masks = masks.astype(np.float32)
                if self.cutbbox and self.mode=="train":
                    if x1_min<self.margin:
                        x1_min=0
                    else:
                        x1_min-=self.margin
                    if y1_min<self.margin:
                        y1_min=0
                    else:
                        y1_min-=self.margin
                    if x1_max>(512-self.margin):
                        x1_max=512
                    else:
                        x1_max+=self.margin
                    if y1_max>(512-self.margin):
                        y1_max=512
                    else:
                        y1_max+=self.margin
                    x1_max=int(x1_max)
                    x1_min=int(x1_min)
                    y1_max=int(y1_max)
                    y1_min=int(y1_min)
                    if (x1_max-x1_min)>(y1_max-y1_min):
                        a=x1_max-x1_min
                        b=y1_max-y1_min
                        remain=a-b
                        while remain!=0:
                            if remain>=2:
                                if y1_min!=0 and y1_max!=512:
                                    y1_min-=1
                                    y1_max+=1
                                elif y1_min==0 and y1_max!=512:
                                    y1_max+=2
                                elif y1_min!=0 and y1_max==512:
                                    y1_min-=2
                                remain-=2
                            elif remain==1:
                                if y1_min!=0 and y1_max!=512:
                                    y1_min-=1
                                elif y1_min==0 and y1_max!=512:
                                    y1_max+=1
                                elif y1_min!=0 and y1_max==512:
                                    y1_min-=1
                                remain-=1
                    elif (x1_max-x1_min)<(y1_max-y1_min):
                        a=x1_max-x1_min
                        b=y1_max-y1_min
                        remain=b-a
                        while remain!=0:
                            if remain>=2:
                                if x1_min!=0 and x1_max!=512:
                                    x1_min-=1
                                    x1_max+=1
                                elif x1_min==0 and x1_max!=512:
                                    x1_max+=2
                                elif x1_min!=0 and x1_max==512:
                                    x1_min-=2
                                remain-=2
                            elif remain==1:
                                if x1_min!=0 and x1_max!=512:
                                    x1_min-=1
                                elif x1_min==0 and x1_max!=512:
                                    x1_max+=1
                                elif x1_min!=0 and x1_max==512:
                                    x1_min-=1
                                remain-=1

                    trfm=A.Compose([
                                    A.Crop(x_min=int(x1_min), y_min=int(y1_min), x_max=int(x1_max), y_max=int(y1_max),always_apply=False, p=1.0),
                                    A.Resize(512,512,p=1.0)
                                    ])
                    images,masks = trfm(image=images, mask=masks)['image'],trfm(image=images, mask=masks)['mask']
                trfm=A.Compose([
                                A.Rotate((30,30),p=1.0),
                                A.HorizontalFlip(p=1.0)
                                ])
                images,masks = trfm(image=images, mask=masks)['image'],trfm(image=images, mask=masks)['mask']
                if self.transform is not None:
                    images,masks = self.transform(image=images, mask=masks)
                
                return images, masks, image_infos

            if self.mode == 'test':
                # transform -> albumentations 라이브러리 활용
                if self.transform is not None:
                    images = self.transform(image=images)
    
                return images, image_infos
        
        elif self.mode!='test'and index>=(len(self.coco.getImgIds())+len(self.plus_list)+len(self.plus_list)+len(self.plus_list)):
            idx=index-len(self.coco.getImgIds())-len(self.plus_list)-len(self.plus_list)-len(self.plus_list)
            image_id = self.coco.getImgIds(imgIds=idx)   
            # getImgIds 는 imgIds를 입력해주면 image_id 를 가져와준다. (여기서는 이미지id 가 0부터 시작해서 따로 사용하지 않고 index 를 바로 넣어줘도 되지만 이미지 id 번호가 띄엄띄엄 저장되어져있는경우는 이렇게 지정해주어야한다.)
            image_infos = self.coco.loadImgs(image_id)[0]
            dataset_path = '/opt/ml/input/data'
            images = cv2.imread(os.path.join(dataset_path, image_infos['file_name'])) # dataset_path/file_name
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
            if (self.mode in ('train', 'val')):
                ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
                anns = self.coco.loadAnns(ann_ids)
                cat_ids = self.coco.getCatIds()
                cats = self.coco.loadCats(cat_ids)
                masks = np.zeros((image_infos["height"], image_infos["width"]))
                if self.cutbbox:
                    x1_min=512
                    y1_min=512
                    x1_max=0
                    y1_max=0
                for i in range(len(anns)):
                    if self.cutbbox:
                        xmin,ymin,width,height=anns[i]['bbox']
                        xmax=xmin+width
                        ymax=ymin+height
                        if x1_min>xmin:
                            x1_min=xmin
                        if y1_min>ymin:
                            y1_min=ymin
                        if x1_max<xmax:
                            x1_max=xmax
                        if y1_max<ymax:
                            y1_max=ymax
                    className = self.get_classname(anns[i]['category_id'], cats)
                    # 그 카테고리 이름에 해당하는 id 값을 가져온다 (단 여기서는 category_names(0 background가 추가된 리스트) 의 인덱스를 가져오는것이다)
                    pixel_value = self.category_names.index(className)
                    # annToMask 는 annotation값을 넣어주면 그 annotation 값에 해당하게 0과 1로 변경시켜준다.(여기서 여러개의 카테고리가 있을수 없다. 하나의 annotation 은 이미지 하나와 카테고리 id 하나만 지정하기 때문이다. 다만 객체는 여러개가 있을수가 있다. 그거를 0과 1로 만 표현해준것이다.)
                    masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
                masks = masks.astype(np.float32)
                if self.cutbbox and self.mode=="train":
                    if x1_min<self.margin:
                        x1_min=0
                    else:
                        x1_min-=self.margin
                    if y1_min<self.margin:
                        y1_min=0
                    else:
                        y1_min-=self.margin
                    if x1_max>(512-self.margin):
                        x1_max=512
                    else:
                        x1_max+=self.margin
                    if y1_max>(512-self.margin):
                        y1_max=512
                    else:
                        y1_max+=self.margin
                    x1_max=int(x1_max)
                    x1_min=int(x1_min)
                    y1_max=int(y1_max)
                    y1_min=int(y1_min)
                    if (x1_max-x1_min)>(y1_max-y1_min):
                        a=x1_max-x1_min
                        b=y1_max-y1_min
                        remain=a-b
                        while remain!=0:
                            if remain>=2:
                                if y1_min!=0 and y1_max!=512:
                                    y1_min-=1
                                    y1_max+=1
                                elif y1_min==0 and y1_max!=512:
                                    y1_max+=2
                                elif y1_min!=0 and y1_max==512:
                                    y1_min-=2
                                remain-=2
                            elif remain==1:
                                if y1_min!=0 and y1_max!=512:
                                    y1_min-=1
                                elif y1_min==0 and y1_max!=512:
                                    y1_max+=1
                                elif y1_min!=0 and y1_max==512:
                                    y1_min-=1
                                remain-=1
                    elif (x1_max-x1_min)<(y1_max-y1_min):
                        a=x1_max-x1_min
                        b=y1_max-y1_min
                        remain=b-a
                        while remain!=0:
                            if remain>=2:
                                if x1_min!=0 and x1_max!=512:
                                    x1_min-=1
                                    x1_max+=1
                                elif x1_min==0 and x1_max!=512:
                                    x1_max+=2
                                elif x1_min!=0 and x1_max==512:
                                    x1_min-=2
                                remain-=2
                            elif remain==1:
                                if x1_min!=0 and x1_max!=512:
                                    x1_min-=1
                                elif x1_min==0 and x1_max!=512:
                                    x1_max+=1
                                elif x1_min!=0 and x1_max==512:
                                    x1_min-=1
                                remain-=1

                    trfm=A.Compose([
                                    A.Crop(x_min=int(x1_min), y_min=int(y1_min), x_max=int(x1_max), y_max=int(y1_max),always_apply=False, p=1.0),
                                    A.Resize(512,512,p=1.0)
                                    ])
                    images,masks = trfm(image=images, mask=masks)['image'],trfm(image=images, mask=masks)['mask']
                trfm=A.Compose([A.HorizontalFlip(p=1.0)
                                ])
                images,masks = trfm(image=images, mask=masks)['image'],trfm(image=images, mask=masks)['mask']
                if self.transform is not None:
                    images,masks = self.transform(image=images, mask=masks)
                
                return images, masks, image_infos

            if self.mode == 'test':
                # transform -> albumentations 라이브러리 활용
                if self.transform is not None:
                    images = self.transform(image=images)
    
                return images, image_infos

        elif self.mode!='test'and index>=(len(self.coco.getImgIds())+len(self.plus_list)+len(self.plus_list)):
            idx=index-len(self.coco.getImgIds())-len(self.plus_list)-len(self.plus_list)
            image_infos = self.coco.loadImgs(self.plus_list[idx])[0]
            dataset_path = '/opt/ml/input/data'
            images = cv2.imread(os.path.join(dataset_path, image_infos['file_name'])) # dataset_path/file_name
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
            if (self.mode in ('train', 'val')):
                ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
                anns = self.coco.loadAnns(ann_ids)
                cat_ids = self.coco.getCatIds()
                cats = self.coco.loadCats(cat_ids)
                masks = np.zeros((image_infos["height"], image_infos["width"]))
                if self.cutbbox:
                    x1_min=512
                    y1_min=512
                    x1_max=0
                    y1_max=0
                for i in range(len(anns)):
                    if self.cutbbox:
                        xmin,ymin,width,height=anns[i]['bbox']
                        xmax=xmin+width
                        ymax=ymin+height
                        if x1_min>xmin:
                            x1_min=xmin
                        if y1_min>ymin:
                            y1_min=ymin
                        if x1_max<xmax:
                            x1_max=xmax
                        if y1_max<ymax:
                            y1_max=ymax
                    className = self.get_classname(anns[i]['category_id'], cats)
                    # 그 카테고리 이름에 해당하는 id 값을 가져온다 (단 여기서는 category_names(0 background가 추가된 리스트) 의 인덱스를 가져오는것이다)
                    pixel_value = self.category_names.index(className)
                    # annToMask 는 annotation값을 넣어주면 그 annotation 값에 해당하게 0과 1로 변경시켜준다.(여기서 여러개의 카테고리가 있을수 없다. 하나의 annotation 은 이미지 하나와 카테고리 id 하나만 지정하기 때문이다. 다만 객체는 여러개가 있을수가 있다. 그거를 0과 1로 만 표현해준것이다.)
                    masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
                masks = masks.astype(np.float32)
                if self.cutbbox and self.mode=="train":
                    if x1_min<self.margin:
                        x1_min=0
                    else:
                        x1_min-=self.margin
                    if y1_min<self.margin:
                        y1_min=0
                    else:
                        y1_min-=self.margin
                    if x1_max>(512-self.margin):
                        x1_max=512
                    else:
                        x1_max+=self.margin
                    if y1_max>(512-self.margin):
                        y1_max=512
                    else:
                        y1_max+=self.margin
                    x1_max=int(x1_max)
                    x1_min=int(x1_min)
                    y1_max=int(y1_max)
                    y1_min=int(y1_min)
                    if (x1_max-x1_min)>(y1_max-y1_min):
                        a=x1_max-x1_min
                        b=y1_max-y1_min
                        remain=a-b
                        while remain!=0:
                            if remain>=2:
                                if y1_min!=0 and y1_max!=512:
                                    y1_min-=1
                                    y1_max+=1
                                elif y1_min==0 and y1_max!=512:
                                    y1_max+=2
                                elif y1_min!=0 and y1_max==512:
                                    y1_min-=2
                                remain-=2
                            elif remain==1:
                                if y1_min!=0 and y1_max!=512:
                                    y1_min-=1
                                elif y1_min==0 and y1_max!=512:
                                    y1_max+=1
                                elif y1_min!=0 and y1_max==512:
                                    y1_min-=1
                                remain-=1
                    elif (x1_max-x1_min)<(y1_max-y1_min):
                        a=x1_max-x1_min
                        b=y1_max-y1_min
                        remain=b-a
                        while remain!=0:
                            if remain>=2:
                                if x1_min!=0 and x1_max!=512:
                                    x1_min-=1
                                    x1_max+=1
                                elif x1_min==0 and x1_max!=512:
                                    x1_max+=2
                                elif x1_min!=0 and x1_max==512:
                                    x1_min-=2
                                remain-=2
                            elif remain==1:
                                if x1_min!=0 and x1_max!=512:
                                    x1_min-=1
                                elif x1_min==0 and x1_max!=512:
                                    x1_max+=1
                                elif x1_min!=0 and x1_max==512:
                                    x1_min-=1
                                remain-=1

                    trfm=A.Compose([
                                    A.Crop(x_min=int(x1_min), y_min=int(y1_min), x_max=int(x1_max), y_max=int(y1_max),always_apply=False, p=1.0),
                                    A.Resize(512,512,p=1.0)
                                    ])
                    images,masks = trfm(image=images, mask=masks)['image'],trfm(image=images, mask=masks)['mask']
                trfm=A.Compose([
                                A.Rotate((45,45),p=1.0)
                                ])
                images,masks = trfm(image=images, mask=masks)['image'],trfm(image=images, mask=masks)['mask']
                if self.transform is not None:
                    images,masks = self.transform(image=images, mask=masks)
                
                return images, masks, image_infos

            if self.mode == 'test':
                # transform -> albumentations 라이브러리 활용
                if self.transform is not None:
                    images = self.transform(image=images)
    
                return images, image_infos


        elif self.mode!='test'and index>=(len(self.coco.getImgIds())+len(self.plus_list)):
            idx=index-len(self.coco.getImgIds())-len(self.plus_list)
            image_infos = self.coco.loadImgs(self.plus_list[idx])[0]
            dataset_path = '/opt/ml/input/data'
            images = cv2.imread(os.path.join(dataset_path, image_infos['file_name'])) # dataset_path/file_name
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
            if (self.mode in ('train', 'val')):
                ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
                anns = self.coco.loadAnns(ann_ids)
                cat_ids = self.coco.getCatIds()
                cats = self.coco.loadCats(cat_ids)
                masks = np.zeros((image_infos["height"], image_infos["width"]))
                if self.cutbbox:
                    x1_min=512
                    y1_min=512
                    x1_max=0
                    y1_max=0
                for i in range(len(anns)):
                    if self.cutbbox:
                        xmin,ymin,width,height=anns[i]['bbox']
                        xmax=xmin+width
                        ymax=ymin+height
                        if x1_min>xmin:
                            x1_min=xmin
                        if y1_min>ymin:
                            y1_min=ymin
                        if x1_max<xmax:
                            x1_max=xmax
                        if y1_max<ymax:
                            y1_max=ymax
                    className = self.get_classname(anns[i]['category_id'], cats)
                    # 그 카테고리 이름에 해당하는 id 값을 가져온다 (단 여기서는 category_names(0 background가 추가된 리스트) 의 인덱스를 가져오는것이다)
                    pixel_value = self.category_names.index(className)
                    # annToMask 는 annotation값을 넣어주면 그 annotation 값에 해당하게 0과 1로 변경시켜준다.(여기서 여러개의 카테고리가 있을수 없다. 하나의 annotation 은 이미지 하나와 카테고리 id 하나만 지정하기 때문이다. 다만 객체는 여러개가 있을수가 있다. 그거를 0과 1로 만 표현해준것이다.)
                    masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
                masks = masks.astype(np.float32)
                if self.cutbbox and self.mode=="train":
                    if x1_min<self.margin:
                        x1_min=0
                    else:
                        x1_min-=self.margin
                    if y1_min<self.margin:
                        y1_min=0
                    else:
                        y1_min-=self.margin
                    if x1_max>(512-self.margin):
                        x1_max=512
                    else:
                        x1_max+=self.margin
                    if y1_max>(512-self.margin):
                        y1_max=512
                    else:
                        y1_max+=self.margin
                    x1_max=int(x1_max)
                    x1_min=int(x1_min)
                    y1_max=int(y1_max)
                    y1_min=int(y1_min)
                    if (x1_max-x1_min)>(y1_max-y1_min):
                        a=x1_max-x1_min
                        b=y1_max-y1_min
                        remain=a-b
                        while remain!=0:
                            if remain>=2:
                                if y1_min!=0 and y1_max!=512:
                                    y1_min-=1
                                    y1_max+=1
                                elif y1_min==0 and y1_max!=512:
                                    y1_max+=2
                                elif y1_min!=0 and y1_max==512:
                                    y1_min-=2
                                remain-=2
                            elif remain==1:
                                if y1_min!=0 and y1_max!=512:
                                    y1_min-=1
                                elif y1_min==0 and y1_max!=512:
                                    y1_max+=1
                                elif y1_min!=0 and y1_max==512:
                                    y1_min-=1
                                remain-=1
                    elif (x1_max-x1_min)<(y1_max-y1_min):
                        a=x1_max-x1_min
                        b=y1_max-y1_min
                        remain=b-a
                        while remain!=0:
                            if remain>=2:
                                if x1_min!=0 and x1_max!=512:
                                    x1_min-=1
                                    x1_max+=1
                                elif x1_min==0 and x1_max!=512:
                                    x1_max+=2
                                elif x1_min!=0 and x1_max==512:
                                    x1_min-=2
                                remain-=2
                            elif remain==1:
                                if x1_min!=0 and x1_max!=512:
                                    x1_min-=1
                                elif x1_min==0 and x1_max!=512:
                                    x1_max+=1
                                elif x1_min!=0 and x1_max==512:
                                    x1_min-=1
                                remain-=1

                    trfm=A.Compose([
                                    A.Crop(x_min=int(x1_min), y_min=int(y1_min), x_max=int(x1_max), y_max=int(y1_max),always_apply=False, p=1.0),
                                    A.Resize(512,512,p=1.0)
                                    ])
                    images,masks = trfm(image=images, mask=masks)['image'],trfm(image=images, mask=masks)['mask']
                trfm=A.Compose([
                                A.Rotate((15,15),p=1.0)
                                ])
                images,masks = trfm(image=images, mask=masks)['image'],trfm(image=images, mask=masks)['mask']
                if self.transform is not None:
                    images,masks = self.transform(image=images, mask=masks)
                
                return images, masks, image_infos

        elif self.mode!='test'and index>=len(self.coco.getImgIds()):
            idx=index-len(self.coco.getImgIds())
            image_infos = self.coco.loadImgs(self.plus_list[idx])[0]
            dataset_path = '/opt/ml/input/data'
            images = cv2.imread(os.path.join(dataset_path, image_infos['file_name'])) # dataset_path/file_name
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
            if (self.mode in ('train', 'val')):
                ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
                anns = self.coco.loadAnns(ann_ids)
                cat_ids = self.coco.getCatIds()
                cats = self.coco.loadCats(cat_ids)
                masks = np.zeros((image_infos["height"], image_infos["width"]))
                if self.cutbbox:
                    x1_min=512
                    y1_min=512
                    x1_max=0
                    y1_max=0
                for i in range(len(anns)):
                    if self.cutbbox:
                        xmin,ymin,width,height=anns[i]['bbox']
                        xmax=xmin+width
                        ymax=ymin+height
                        if x1_min>xmin:
                            x1_min=xmin
                        if y1_min>ymin:
                            y1_min=ymin
                        if x1_max<xmax:
                            x1_max=xmax
                        if y1_max<ymax:
                            y1_max=ymax
                    className = self.get_classname(anns[i]['category_id'], cats)
                    # 그 카테고리 이름에 해당하는 id 값을 가져온다 (단 여기서는 category_names(0 background가 추가된 리스트) 의 인덱스를 가져오는것이다)
                    pixel_value = self.category_names.index(className)
                    # annToMask 는 annotation값을 넣어주면 그 annotation 값에 해당하게 0과 1로 변경시켜준다.(여기서 여러개의 카테고리가 있을수 없다. 하나의 annotation 은 이미지 하나와 카테고리 id 하나만 지정하기 때문이다. 다만 객체는 여러개가 있을수가 있다. 그거를 0과 1로 만 표현해준것이다.)
                    masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
                masks = masks.astype(np.float32)
                if self.cutbbox and self.mode=="train":
                    if x1_min<self.margin:
                        x1_min=0
                    else:
                        x1_min-=self.margin
                    if y1_min<self.margin:
                        y1_min=0
                    else:
                        y1_min-=self.margin
                    if x1_max>(512-self.margin):
                        x1_max=512
                    else:
                        x1_max+=self.margin
                    if y1_max>(512-self.margin):
                        y1_max=512
                    else:
                        y1_max+=self.margin
                    x1_max=int(x1_max)
                    x1_min=int(x1_min)
                    y1_max=int(y1_max)
                    y1_min=int(y1_min)
                    if (x1_max-x1_min)>(y1_max-y1_min):
                        a=x1_max-x1_min
                        b=y1_max-y1_min
                        remain=a-b
                        while remain!=0:
                            if remain>=2:
                                if y1_min!=0 and y1_max!=512:
                                    y1_min-=1
                                    y1_max+=1
                                elif y1_min==0 and y1_max!=512:
                                    y1_max+=2
                                elif y1_min!=0 and y1_max==512:
                                    y1_min-=2
                                remain-=2
                            elif remain==1:
                                if y1_min!=0 and y1_max!=512:
                                    y1_min-=1
                                elif y1_min==0 and y1_max!=512:
                                    y1_max+=1
                                elif y1_min!=0 and y1_max==512:
                                    y1_min-=1
                                remain-=1
                    elif (x1_max-x1_min)<(y1_max-y1_min):
                        a=x1_max-x1_min
                        b=y1_max-y1_min
                        remain=b-a
                        while remain!=0:
                            if remain>=2:
                                if x1_min!=0 and x1_max!=512:
                                    x1_min-=1
                                    x1_max+=1
                                elif x1_min==0 and x1_max!=512:
                                    x1_max+=2
                                elif x1_min!=0 and x1_max==512:
                                    x1_min-=2
                                remain-=2
                            elif remain==1:
                                if x1_min!=0 and x1_max!=512:
                                    x1_min-=1
                                elif x1_min==0 and x1_max!=512:
                                    x1_max+=1
                                elif x1_min!=0 and x1_max==512:
                                    x1_min-=1
                                remain-=1

                    trfm=A.Compose([
                                    A.Crop(x_min=int(x1_min), y_min=int(y1_min), x_max=int(x1_max), y_max=int(y1_max),always_apply=False, p=1.0),
                                    A.Resize(512,512,p=1.0)
                                    ])
                    images,masks = trfm(image=images, mask=masks)['image'],trfm(image=images, mask=masks)['mask']
                trfm=A.Compose([
                                A.Rotate((30,30),p=1.0)
                                ])
                images,masks = trfm(image=images, mask=masks)['image'],trfm(image=images, mask=masks)['mask']
                if self.transform is not None:
                    images,masks = self.transform(image=images, mask=masks)
                
                return images, masks, image_infos
            
            if self.mode == 'test':
                # transform -> albumentations 라이브러리 활용
                if self.transform is not None:
                    images = self.transform(image=images)
    
                return images, image_infos        

        else:
            image_id = self.coco.getImgIds(imgIds=index)   
            # getImgIds 는 imgIds를 입력해주면 image_id 를 가져와준다. (여기서는 이미지id 가 0부터 시작해서 따로 사용하지 않고 index 를 바로 넣어줘도 되지만 이미지 id 번호가 띄엄띄엄 저장되어져있는경우는 이렇게 지정해주어야한다.)
            image_infos = self.coco.loadImgs(image_id)[0]   
            # loadImgs 는 image id 에 해당하는 이미지 정보들을 가져와준다.[]때문에 [0]을 해주는것이다. 
            # [{'width':3024,"height":2268,"file_name":"batch_06/0001.jpg","license":0,"flickr_url":null,"coco_url":null,"date_captured":"2020-12-28 11:04:34","id":0}]
            dataset_path = '/opt/ml/input/data'
            # cv2 를 활용하여 image 불러오기
            images = cv2.imread(os.path.join(dataset_path, image_infos['file_name'])) # dataset_path/file_name
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32) # cv 로이미지를 읽어오면 BGR로 읽어져와서 RGB로 바꿔줘야한다. 바꿔준후에 넘파이 형식으로 또 바꿔줘야한다.

            
            if (self.mode in ('train', 'val')): # train하고 val 인경우 (정답이 있는 경우)
                ann_ids = self.coco.getAnnIds(imgIds=image_infos['id']) 
                # getAnnIds 는 image_id 를 입력해주면 그 이미지 한장에 들어있는 모든 anotations id(카테고리 id 도 아니고 image id 도 아니고 annotation id 이다.(0번~(annotation개수-1))를 가져오게 된다.
                # ann_ids 는 [] 리스트 형태로 반환해준다. 한이미지 안에 annotation 이 여러개가 있을수 있기 때문이다. [10],[30,43] 이런식으로 저장된다.
                anns = self.coco.loadAnns(ann_ids) 
                # loadAnns 는 ann_ids(리스트)를 넣어주면 그 id 에 해당하는 annotations 를 반환해준다 . id 1개당 무조건 segmentation 이 한개이다.똑같은 이미지 아이디에 똑같은 카테고리라고 해도 그런게 두개있으면 다음 id 로 넘어간다. 그래서 segmentation하고 bbox 는 무조건 한개 이다.
                # anns =  [{'image_id':0,"category_id":6,"segmentation":[[1731,619,1730,620,....]],"area":606046,"bbox":[831.0,619.0,1379.0,764.0],"iscrowd":0,"id":0},{'image_id':0,"categroy_id":1,"segmentation":[[516,0,517,1,....]],"area":~,"bbox":[~],"iscrowd":~,"id":1},....]
                
                # Load the categories in a variable
                cat_ids = self.coco.getCatIds() 
                # getCatIds() 를 해주면 모든 json 파일에 저장된 모든 category id 를 가져오게 된다.[0,1,2,3,4,5,6,7,8,9,10]
                cats = self.coco.loadCats(cat_ids) 
                # loadCats(모든카테고리id들)을 넣어주면 카테고리 id 에 해당하는 category 정보들을 주게 된다.
                # cats = [{"id":0,"name":"UNKNOWN","supercategory":"UNKNOWN"},{"id":1,"name":"General trash","supercategory":"General trash"},....]
                
                # masks : size가 (height x width)인 2D
                # 각각의 pixel 값에는 "category id + 1" 할당
                # Background = 0
                masks = np.zeros((image_infos["height"], image_infos["width"]))
                # Unknown = 1, General trash = 2, ... , Cigarette = 11
                
                # 한이미지내의 anns 개수만큼 반복해준다.
                if self.cutbbox:
                    x1_min=512
                    y1_min=512
                    x1_max=0
                    y1_max=0
                for i in range(len(anns)):
                    if self.cutbbox:
                        xmin,ymin,width,height=anns[i]['bbox']
                        xmax=xmin+width
                        ymax=ymin+height
                        if x1_min>xmin:
                            x1_min=xmin
                        if y1_min>ymin:
                            y1_min=ymin
                        if x1_max<xmax:
                            x1_max=xmax
                        if y1_max<ymax:
                            y1_max=ymax

                    # 카테고리 id 에 해당하는 카테고리 이름(className)을 가져온다.
                    className = self.get_classname(anns[i]['category_id'], cats)
                    # 그 카테고리 이름에 해당하는 id 값을 가져온다 (단 여기서는 category_names(0 background가 추가된 리스트) 의 인덱스를 가져오는것이다)
                    pixel_value = self.category_names.index(className)
                    # annToMask 는 annotation값을 넣어주면 그 annotation 값에 해당하게 0과 1로 변경시켜준다.(여기서 여러개의 카테고리가 있을수 없다. 하나의 annotation 은 이미지 하나와 카테고리 id 하나만 지정하기 때문이다. 다만 객체는 여러개가 있을수가 있다. 그거를 0과 1로 만 표현해준것이다.)
                    masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
                # mask 를 넘파이float32형태로 변경시켜준다.
                masks = masks.astype(np.float32)

                if self.cutbbox and self.mode=="train":
                    if x1_min<self.margin:
                        x1_min=0
                    else:
                        x1_min-=self.margin
                    if y1_min<self.margin:
                        y1_min=0
                    else:
                        y1_min-=self.margin
                    if x1_max>(512-self.margin):
                        x1_max=512
                    else:
                        x1_max+=self.margin
                    if y1_max>(512-self.margin):
                        y1_max=512
                    else:
                        y1_max+=self.margin
                    x1_max=int(x1_max)
                    x1_min=int(x1_min)
                    y1_max=int(y1_max)
                    y1_min=int(y1_min)
                    if (x1_max-x1_min)>(y1_max-y1_min):
                        a=x1_max-x1_min
                        b=y1_max-y1_min
                        remain=a-b
                        while remain!=0:
                            if remain>=2:
                                if y1_min!=0 and y1_max!=512:
                                    y1_min-=1
                                    y1_max+=1
                                elif y1_min==0 and y1_max!=512:
                                    y1_max+=2
                                elif y1_min!=0 and y1_max==512:
                                    y1_min-=2
                                remain-=2
                            elif remain==1:
                                if y1_min!=0 and y1_max!=512:
                                    y1_min-=1
                                elif y1_min==0 and y1_max!=512:
                                    y1_max+=1
                                elif y1_min!=0 and y1_max==512:
                                    y1_min-=1
                                remain-=1
                    elif (x1_max-x1_min)<(y1_max-y1_min):
                        a=x1_max-x1_min
                        b=y1_max-y1_min
                        remain=b-a
                        while remain!=0:
                            if remain>=2:
                                if x1_min!=0 and x1_max!=512:
                                    x1_min-=1
                                    x1_max+=1
                                elif x1_min==0 and x1_max!=512:
                                    x1_max+=2
                                elif x1_min!=0 and x1_max==512:
                                    x1_min-=2
                                remain-=2
                            elif remain==1:
                                if x1_min!=0 and x1_max!=512:
                                    x1_min-=1
                                elif x1_min==0 and x1_max!=512:
                                    x1_max+=1
                                elif x1_min!=0 and x1_max==512:
                                    x1_min-=1
                                remain-=1

                    trfm=A.Compose([
                                    A.Crop(x_min=int(x1_min), y_min=int(y1_min), x_max=int(x1_max), y_max=int(y1_max),always_apply=False, p=1.0),
                                    A.Resize(512,512,p=1.0)
                                    ])
                    images,masks = trfm(image=images, mask=masks)['image'],trfm(image=images, mask=masks)['mask']

                # transform -> albumentations 라이브러리 활용
                if self.transform is not None:
                    images,masks = self.transform(image=images, mask=masks)
                
                return images, masks, image_infos
            
            if self.mode == 'test':
                # transform -> albumentations 라이브러리 활용
                if self.transform is not None:
                    images = self.transform(image=images)
    
                return images, image_infos
    
    # 전체 데이터셋 길이는 전체 이미지 개수이다.
    def __len__(self) -> int:
        if (self.mode in ('train', 'val')):
            return 2*(len(self.coco.getImgIds())+len(self.plus_list)+len(self.plus_list)+len(self.plus_list))+len(self.pseudodata)
        # 전체 dataset의 size를 return
        else:
            return len(self.coco.getImgIds())