# Here I showed augmentation  of images with and without bounding boxes using PyTorch, Tensorflow, Albumentations and Imgaug. 
# Enjoy !
# You can find bbox augmentation outputs in imgaug and albumentations folders. I looked bbox augmentation on tensorflow and PyTorch but I couldn't find any. May be there is not but I am not sure. I think someone who wants to create CV models should use PyTorch+Albumentations.
**
Augmentation is a very effective method to make your method more robust.  As a matter of fact if you try to create models that Works quite well you should consider to use data augmentation. 
#
First  install all of the requirements.
***
***
`pip install -r  requirements.txt`

#
***
**_"Any fool can know. The point is to understand." Albert Einstein_**
****
****
 

#





# Using imgaug

_Actually top left one the original one but I set titles of all images according to their augmentation. I automated the process and I told myself if I set probability  0 for fliplr it will return original image. Sorry I am too lazy to code if else for naming._ 

![Rotate](https://github.com/abdullahbas/Data-Augmentation/blob/main/augmented.png?raw=true)
##
# Using imgaug 2 

_Same code but different output because of the probabilities._





![Channel Shuff](https://github.com/abdullahbas/Data-Augmentation/blob/main/augmented2.png?raw=true)

##
# Original Images

![Tensorflow](https://github.com/abdullahbas/Data-Augmentation/blob/main/TFOrig.png?raw=true)
#
# Using Tensorflow 

_Tensorflow has tf.image  function that can be used for augmentations._

![Tensorflow](https://github.com/abdullahbas/Data-Augmentation/blob/main/TFAugmented.png?raw=true)

#

# Using Tensorflow (2)

_On training you can consider these images as epochs. As you can see, although you have same original image you will end up with completely different images._

![Tensorflow 2 ](https://github.com/abdullahbas/Data-Augmentation/blob/main/TFAugmented2.png?raw=true)
#
# Using PyTorch 

_On training you can consider these images as epochs. As you can see, although you have same original image you will end up with completely different images._

![PyTorch ](https://github.com/abdullahbas/Data-Augmentation/blob/main/PyTorchAugmented.png?raw=true)
#
# Using PyTorch (2)


![PyTorch2 ](https://github.com/abdullahbas/Data-Augmentation/blob/main/PyTorchAugmented2.png?raw=true)


# Using Albumentations -- I think the best one especially when you work on object detection task. 

_You can add augmentation to your pipeline easily with albumentations. Actually I decided to write new library because of not-adequate built-in functions for bbox augmentation before I met albumentations. You can use it in dataloader of PyTorch, in generators of Tensorflow and also can combine with other augmentations from other frameworks. Thanks albumentations you nailed it !_

![Albumentations ](https://github.com/abdullahbas/Data-Augmentation/blob/main/AlbumentationsAugmented1.png?raw=true)
#
# Using Albumentations (2)


![Albumentations ](https://github.com/abdullahbas/Data-Augmentation/blob/main/AlbumentationsAugmented2.png?raw=true)











