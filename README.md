# Here I showed augmentation  of images with and without bounding boxes using PyTorch, Tensorflow, Albumentations and Imgaug. 
# Enjoy !

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












