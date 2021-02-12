# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 17:28:30 2021

@author: trabz
"""
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from xmlParser import Parser
import glob
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
path= 'Image A*/train/*.xml'
import cv2



seq = iaa.Sequential([
    iaa.Fliplr(p=0),# basically this is original one
    iaa.Sometimes(0.05,(iaa.Crop(px=(22, 45),keep_size=True))), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Sometimes(0.5,(iaa.Fliplr(1))), # horizontally flip 50% of the images
    iaa.Sometimes(0.02,iaa.GaussianBlur(sigma=(5, 7.0))), # blur images with a sigma of 0 to 3.0
    iaa.Sometimes(0.02 ,iaa.ImpulseNoise(p=(0.6,1))),
    iaa.Sometimes(0.02 ,iaa.EdgeDetect(alpha=(0.09,1))),
    #iaa.AddToBrightness(add=(100,124)),
    iaa.Sometimes(0.02 ,iaa.Canny(alpha=(0.8,0.9))),
    iaa.Sometimes(0.5 ,iaa.Grayscale(alpha=1.00)),
    iaa.Sometimes(0.5 ,iaa.ChannelShuffle(p=1)),
    #iaa.Sometimes(0.02 ,(iaa.geometric.Affine( scale=2,rotate=22,order=1))),
    iaa.Sometimes(0.5 ,iaa.Cartoon(blur_ksize=(11,13))),
    iaa.Sometimes(0.02 ,iaa.CenterCropToAspectRatio(1)),
    iaa.Sometimes(0.02 ,iaa.CenterCropToFixedSize(100,100)),
    iaa.Sometimes(0.12 ,iaa.ChangeColorTemperature(kelvin=(2222,3333))),
    #iaa.segmentation(),
    iaa.Sometimes(0.12 ,iaa.CLAHE(clip_limit=(4,8))),
    iaa.Sometimes(0.8 ,iaa.Rotate(rotate=(-90,90),order=1))
])

plt.figure(figsize=(12,12))


ls=glob.glob(path)
res=[]
for idx,l in enumerate(ls):
    res.append(Parser.myType(l,idx,classes=['bird','zebra']))


def our_generator(res):
    for i in res:
      img= cv2.imread(ls[i['image_id']].replace('.xml','.jpg'))
      result =i
      yield img,result

generator=our_generator(res)


# image,result=next(iter(generator))
# image_aug,bbs=seq(image=image,bounding_boxes=result['bbs'])
# for bb in bbs:
#     image_aug = bb.draw_on_image(image_aug, size=2, color=[0, 0, 255])
# cv2.imshow('dd',np.hstack((image,image_aug)))

for idx,(image,result) in enumerate(generator):

    bbs = BoundingBoxesOnImage(result['bbs'], shape=image.shape)
    image_aug,bbs=seq(image=image,bounding_boxes=bbs)
    for idc,bb in enumerate(bbs):
        image_aug = bb.draw_on_image(image_aug, size=2, color=[0, 0, 255])
        image=result['bbs'][idc].draw_on_image(image,size=2,color=[0,255,0])
    cv2.imshow('dd',np.hstack((image,image_aug)))
    cv2.imwrite(f'imgaugs/{idx}.png',np.hstack((image,image_aug)))

cv2.waitKey(0)
cv2.destroyAllWindows()


