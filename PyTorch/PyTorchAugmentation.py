# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 19:24:17 2021

@author: trabz
"""
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
            
            image_aug=self.transform(image)
        
        return image,image_aug,annotation


def collate_fn(batch):
            return tuple(zip(*batch))


data_transform=transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.ToPILImage(),
    transforms.RandomApply([transforms.CenterCrop(222), transforms.RandomCrop(170)],p=0.01),
   
    
    transforms.RandomAffine((-15,15)),
        

    
    #transforms.Grayscale(),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomApply([transforms.ColorJitter(brightness=15,contrast=12,hue=0.2)],p=0.1),
    # transforms.RandomResizedCrop((200,200)),
    transforms.RandomRotation((-90,90)),
    transforms.RandomPerspective(p=0.1),
    transforms.ToTensor(),
    # transforms.RandomCrop(10),
    transforms.RandomErasing(p=0.1),

    
    # 
    ])






path=path= 'Image A*/train/*.xml'

dataset = dataloader(path,data_transform)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, collate_fn=collate_fn)




plt.figure(figsize=(12,12))
# for idx,(image,imgO,result) in enumerate(data_loader):
    
#     #plt.figure()
#     #plt.imshow(result[0].permute(1,2,0).numpy(),cmap='gray')
#     img=imgO[0].clone().permute(1,2,0).numpy()
#     image=image[0]
#     sz,wd,_=np.array(image.shape)-np.array((img).shape)
#     img2=img
#     img=np.pad((img),((sz//2,sz//2),(wd//2,wd//2),(0,0)))
    
#     try:
#         plt.imsave(f'PyTorch/images/{idx}.png',np.hstack(((img*255).astype('uint8'),image)))
#     except ValueError:
#         img=np.dstack((img,img,img))
#         plt.imsave(f'PyTorch/images/{idx}.png',np.hstack(((img*255).astype('uint8'),image)))
#     plt.axis('off')
#     plt.tight_layout()
#     #plt.imshow(np.dstack((img[:,:,2],img[:,:,1],img[:,:,0])))
#     plt.imshow(image)    
    
#♣plt.savefig('TFOrig.png',dpi=110,transparent=True)

for idx,(image,imgO,result) in enumerate(data_loader):
    
    #plt.figure()
    #plt.imshow(result[0].permute(1,2,0).numpy(),cmap='gray')
    plt.subplot(6,5,idx+1)
    
    plt.axis('off')
    plt.tight_layout()
    #plt.imshow(np.dstack((img[:,:,2],img[:,:,1],img[:,:,0])))
    plt.imshow(imgO[0].permute(1,2,0).numpy())    
    
plt.savefig('PyTorchAugmented2.png',dpi=110,transparent=True)
