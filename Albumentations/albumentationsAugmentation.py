# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 15:26:57 2021

@author: trabz
"""
import albumentations as A

import torch 


from xmlParser import Parser
import glob
import numpy as np
from matplotlib import pyplot as plt
import cv2
import torchvision.transforms as transforms


class dataloader():
    
    def __init__(self,path,transform=None):
        
        self.paths=glob.glob(path)
        self.transform=transform
        
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,idx):
        
        annotation=(Parser.myType(self.paths[idx],idx,classes=['bird','zebra']))
        image=plt.imread(self.paths[annotation['image_id']].replace('xml','jpg'))
        
        if self.transform is not None:
            
            augmented = self.transform(image=image, bboxes=annotation['bbox'], labels=annotation['label'])

        
        return image,augmented,annotation


def collate_fn(batch):
            return tuple(zip(*batch))


bbox_params = A.BboxParams(
  format='pascal_voc', 
  min_area=1, 
  min_visibility=0.5, 
  label_fields=['labels']
)

aug = A.Compose({
    #A.Resize(500, 500,p=0.2),
        A.RGBShift(r_shift_limit=40,g_shift_limit=40,b_shift_limit=40,p=0.04),
        A.RandomBrightness(p=0.01),
        A.RandomContrast(p=0.01),
        A.CLAHE(p=0.02),
        A.ToGray(p=0.4),
        A.Blur(blur_limit=8,p=0.1),
        A.RandomBrightness(p=0.1),
        A.CenterCrop(100, 100,p=0.01),
        A.RandomCrop(222, 222,p=0.1),
        A.HorizontalFlip(p=0.1),
        A.Rotate(limit=(-90, 90),p=0.2),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(),
        
        },bbox_params=bbox_params)


path=path= 'Image A*/train/*.xml'

dataset = dataloader(path,aug)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, collate_fn=collate_fn,shuffle=False)




plt.figure(figsize=(12,12))
for idx,(image,imgO,result) in enumerate(data_loader):
    
    
    imgA=imgO[0]['image']
    image=image[0]
    for idc,bbox in enumerate(imgO[0]['bboxes']):
        xmin,ymin,xmax,ymax=bbox
        xmin,xmax=np.clip([xmin,xmax],5,imgA.shape[0]-5).astype('int')
        ymin,ymax=np.clip([ymin,ymax],5,imgA.shape[1]-5).astype('int')
        
        xpastmin,ypastmin,xpastmax,ypastmax=np.clip((result[0]['bbox'][idc]),0,max(image.shape)-10)
        
        imgA=cv2.rectangle(np.array(imgA),((xmin),(ymin)),((xmax),(ymax))
                           ,color=[0,245,0],thickness=4)
       
        image=cv2.rectangle(image,((xpastmin),(ypastmin)),((xpastmax),(ypastmax))
                           ,color=[112,9,11],thickness=4)
    
    
    
    
    sz,wd,_=np.array(image.shape)-np.array((imgA).shape)
    img2=imgA
    imgA=np.pad((imgA),((sz//2,sz//2),(wd//2,wd//2),(0,0)))
    
    
    plt.imsave(f'Albumentations/images/{idx}.png',np.hstack(((imgA).astype('uint8'),image)))
    plt.axis('off')
    plt.tight_layout()
    #plt.imshow(np.dstack((img[:,:,2],img[:,:,1],img[:,:,0])))





