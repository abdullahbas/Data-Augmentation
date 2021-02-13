This section is about imgaug library

Actually I like all the documentation and the stuffs of this library but there transformation of bounding boxes.

You should change and manipulate the paths. For example I will add xml files and images that I used in this tutorial to the main path. Change path to the correct one to reproduce the result of this work.


#

Main thing of augmentation is probability of performing. When you start to investigate scripts you will see 

< iaa.Sometimes () >

#
This is the code that give us the power to determine probability of performing. Other things are quite understandable in the scripts except xml parsing. I will not dive deep into that part because of the focus of this repo. If you have any questions respect to xml parsing please feel free to ask. 


#
***
**_"Own not buy." Erich Fromm_**
****
****
 

#

_I will show you some outputs of augmentations. All of the augmentations that I used in this repo are present in the scripts._ 
#

#

# Rotate and fliplr

_Green bbox (left) is the gt and the red bbox and image (right) is the augmented image._

![Rotate](https://github.com/abdullahbas/Data-Augmentation/blob/main/imgaugs/0.png?raw=true)
##
# Rotate and Channel Shuffling

.


![Channel Shuff](https://github.com/abdullahbas/Data-Augmentation/blob/main/imgaugs/1.png?raw=true)
#

# Rotate and Cartooning
_I show rotates to demonstrate the bbox success and failures. You can check further by investigating  the images that I uploaded_

![Line Plot](https://github.com/abdullahbas/Data-Augmentation/blob/main/imgaugs/24.png?raw=true
)

