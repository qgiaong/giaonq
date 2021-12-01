---
author_profile: true
title:  "Face Mask Detector"
categories:
  - machine-learning
tags:
  - deep learning
  - machine learning
  - computer vision
header:
  image: /assets/images/facemask.jpg

---

# Face Mask Detection
Face masks play a central role in protecting the health of community against COVID-19. In this project, we study the 
[Face Mask Detection Dataset](https://www.kaggle.com/andrewmvd/face-mask-detection), which contains 853 images of people in many daylife situations. Each image contain annotations that are divided in 3 classes (with mask, without mask, mask worn incorrectly), together with the coresponding bounding boxes in the [PASCAL VOC format](http://host.robots.ox.ac.uk/pascal/VOC/). 

In particular, we will train a neural network for detecting people faces in images and classifying whether they wear face masks correctly. The objective is to adopt deep learning technique to detect if people violating norms of wearing masks in public. We are not training the our neural network from scratch, but employing the pretrained ResNet model and finetune it for our purpose. 





