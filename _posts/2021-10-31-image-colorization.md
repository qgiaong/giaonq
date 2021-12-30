---
author_profile: true
title:  "Image Colorization"
categories:
  - machine-learning
tags:
  - deep learning
  - machine learning
  - computer vision
header:
  image: /assets/images/colors.png
  teaser: /assets/images/colors.png
---
# Introduction
In this blog spot, we are going to tackle a Computer Vision problem called Image Colorization. In general, image colorization refers to converting a grayscale image (1 channel) to a full color image (3 channel). This is a challenging process because colorization is multi-modal, in such a way that a grayscale image can be mapped to several plausible colored images. 


Traditionally, this problem must be done manually and requires enormous attempt of human hardcoding. Instead, deep learning can be used to make this process automatic. We are going to make use of the ability of deep learning models to cature semantic information in images and to build a deep learning model using Pytorch. 

## Background on Colors
RGB format is often used to represent color images due to its simplicity. Each channel of this format represents a single color value, indicating how much Red, Green, Blue a pixel is. The color of the pixel by computing the sum of these 3 channels. The following plot gives an example of these 3 channels. The first image is the red channel, and we see that the red part of the original image is much darker in this channel.

![rgb](https://user-images.githubusercontent.com/43914109/147781992-f0b13a13-4de1-4fe8-ba69-be4a7243b20b.PNG)

Another common color format is the L\*a\*b format. In the L\*a\*b space, we also have 3 values but with different meaning. The first channel, L, represents the lightness of the pixel and contains the image in black-and-white. The *a  and *b values encode how much green-red and yellow-blue each pixel is, respectively. These 3 channels are visualized in the following plot: 
![lab](https://user-images.githubusercontent.com/43914109/147781998-6976ccd4-b091-46eb-966e-3210f668a499.png)

It is common for image colorization task to use L\*a\*b instead of RGB format, since we can separate the grayscale part (the lightness channel) directly. Therefore, we can generate the input data for the model directly and can formulate our problem as to reconstruct the  *a  and *b channel from the L channel.

## Training strategy
As mentioned above, the task is to reconstruct the full-colored image using only the black-and-white image from the L channel. For this purpose, we adopt the pix2pix model provided by the [_**Image-to-Image Translation with Conditional Adversarial Networks**_](https://arxiv.org/abs/1611.07004) paper. 
The original implementation is provided in their [Github](https://github.com/cathmer/pix2pix/).

# Dataset

For training, we create a L\*a\*b dataset using existing images and creating grayscale versions of photos that models must learn to colorize. We make use of the 
[Lanscape Pictures Dataset](https://www.kaggle.com/arnaud58/landscape-pictures) containing  more than 4000 images of real-world landscape scenes. We make the problem a little bit easier, because the multimodality in nature scences is not so strong: sky is often blue and fields are green. This is not always the case: a car in grayscale image can have a red or blue color in real life. First, you need to download and unzip the dataset from Kaggle. We assume for the moment that the data is stored in the `data` folder.

Let's collect all the image paths:
```
path = "/content/data"
paths = np.array(glob.glob(path + "/*.jpg")) 
np.random.seed(SEED)
len(paths)
```


