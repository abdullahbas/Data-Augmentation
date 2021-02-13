# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 00:51:01 2021

@author: trabz
"""

from xmlParser import Parser
import glob
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2

def augmentator(image):
    
    img=image.copy()
    img=random(tf.image.random_brightness(img,max_delta=0.2),img,p=0.4)
    img=random(tf.image.adjust_contrast(img,contrast_factor=2),img)
    img=random(tf.image.adjust_jpeg_quality(img,2),img)
    img=random(tf.image.flip_left_right(img),img)
    img=random(tf.image.flip_up_down(img),img)
    img=random(tf.image.adjust_hue(img,delta=0.4),img,p=0.7)
    img=random(tf.image.random_flip_left_right(img),img)
    img=random(tf.image.rot90(img),img)
    img=random(tf.image.rgb_to_grayscale(img),img,p=0.2)
    img=random(tf.image.central_crop(img,0.5),img)
    
    return img,image


def random(aug,image,p=0.2):
    
    if np.random.random()<p:
        image=aug
    else:
        pass
    return image        


path= 'Image A*/train/*.xml'
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


plt.figure(figsize=(12,12))
for idx,(image,result) in enumerate(generator):
    
        
    plt.subplot(5,6,idx+1)
    img,image=augmentator(image)
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(np.array(img))    
    
plt.savefig('TFAugmented2.png',dpi=110,transparent=True)











