# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 03:02:47 2021

@author: trabz
"""
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

img=plt.imread('bird.jpg')




seq = iaa.Sequential([
    iaa.Fliplr(p=0),# basically this is original one
    iaa.Crop(px=(22, 45),keep_size=False), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(1), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(5, 7.0)), # blur images with a sigma of 0 to 3.0
    iaa.ImpulseNoise(p=(0.6,1)),
    iaa.EdgeDetect(alpha=(0.9,1)),
    #iaa.AddToBrightness(add=(100,124)),
    iaa.Canny(alpha=(0.8,0.9)),
    iaa.Grayscale(alpha=1.00),
    iaa.ChannelShuffle(p=1),
    iaa.geometric.Affine( scale=2,rotate=22, backend='cv2'),
    iaa.Cartoon(blur_ksize=(11,13)),
    iaa.CenterCropToAspectRatio(1),
    iaa.CenterCropToFixedSize(100,100),
    iaa.ChangeColorTemperature(kelvin=(2222,3333)),
    #iaa.segmentation(),
    iaa.CLAHE(clip_limit=(4,8)),
    iaa.Rotate(rotate=(-30,90))
])

plt.figure(figsize=(12,12))

for idx,Augmentor in enumerate(seq):
    # print(1)
    ax=plt.subplot(4,4,idx+1)
    ax.axis('off')
    plt.tight_layout()
    title=str(Augmentor).split('(')[0]
    #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
    #        hspace = 0.2, wspace = 0)
    #plt.margins(0,0)
    ax.set_title(title)
    plt.imshow(Augmentor(image=img))
plt.savefig('augmented.png',dpi=110,transparent=True)


