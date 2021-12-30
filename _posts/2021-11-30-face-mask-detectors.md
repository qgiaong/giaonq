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
  teaser: /assets/images/facemask.jpg
---

Face masks play a central role in protecting the health of community against COVID-19. In this project, we study the 
[Face Mask Detection Dataset](https://www.kaggle.com/andrewmvd/face-mask-detection), which contains 853 images of people in many daylife situations. Each image contain annotations that are divided in 3 classes (with mask, without mask, mask worn incorrectly), together with the coresponding bounding boxes in the [PASCAL VOC format](http://host.robots.ox.ac.uk/pascal/VOC/). 

In particular, we will train a neural network for detecting people faces in images and classifying whether they wear face masks correctly. The objective is to adopt deep learning technique to detect if people violating norms of wearing masks in public. We are not training the our neural network from scratch, but employing the pretrained ResNet model and finetune it for our purpose. 


Let first import the needed packages:
```
import os
import numpy as np
import pandas as pd

import torch
import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from bs4 import BeautifulSoup
```
## Prepare Data
Next, we need some helper function to generate training data from the extracted dataset. We use `BeautifulSoup` package to extract information from the .xml annotation files. Each identified object is assigned an integer label (`"without_masks" = 1, "with_mask" = 2, "mask_weared_incorrect" = 3`), as desired by Pytorch. Besides, the corresponding bounding boxes are also converted to Pytorch format `[xmin, ymin, xmax, ymax]`

```
def generate_box(obj):
    # get bounding box coordinates in pytorch format for a given object
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text) 
    return [xmin, ymin, xmax, ymax]

def generate_label(obj):
    # assign label to object. Note that the label starts from 1, 
    # since label 0 is reserved for the background 
    if obj.find('name').text == "with_mask":
        return 2
    elif obj.find('name').text == "mask_weared_incorrect":
        return 3
    return 1

def generate_target(image_id, file): 
    # generate training target from the annotation file
    with open(file) as f:
        data = f.read()
        # read xml file
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')
        num_objs = len(objects)

        boxes = []
        labels = []
        for i in objects: # extract annotation of each object
            boxes.append(generate_box(i))
            labels.append(generate_label(i))

        # convert all to pytorch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([image_id])
        # save annotation in dictionary for each image
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        
        return target

```
Now we are defining our custom dataset. Our dataset class should inherit from `torch.utils.data.Dataset` and implement the functions `__getitem__` and  `__len__`
```
class MaskDataset(torch.utils.data.Dataset):
  def __init__(self, transforms):
        self.transforms = transforms
        # load all image and annotation files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir("data/images/")))
        self.masks = list(sorted(os.listdir("data/annotations/")))
  def __getitem__(self, idx):
        # load images and masks
        img_path   = os.path.join("data/images/",  self.imgs[idx])
        label_path = os.path.join("data/annotations/",  self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        #Generate Label
        target = generate_target(idx, label_path)
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

  def __len__(self):
        return len(self.imgs)
  ```
  We are curious how the masks are distributed. To findout, we read all labels available in the dataset, and use Counter to get the number of each class
 ```
 all_labels = []
ann_paths =  list(sorted(os.listdir("data/annotations/")))
for mask_path in ann_paths:
   with open(os.path.join("data/annotations/", mask_path)) as f:
      data = f.read()
      # read xml file
      soup = BeautifulSoup(data, 'xml')
      objects = soup.find_all('object')
      num_objs = len(objects)
      labels = []
      for i in objects:
          all_labels.append(generate_label(i))
 ```
 
  ```
  from collections import Counter

class_names = ["without_masks", "with_mask", "mask_weared_incorrect"]
values = Counter(all_labels).values()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize =(14,6))
background_color = '#faf9f4'
ax1.set_facecolor(background_color)
ax2.set_facecolor(background_color) 
ax1.pie(values,wedgeprops=dict(width=0.3, edgecolor='w') ,
        labels=class_names, radius=1, startangle = 180, autopct='%1.2f%%')

ax2 = plt.bar(class_names, list(values),
              color ='maroon',width = 0.4)
   
plt.show()
   ```
 We see that most annotations are about people wearing mask. 
 ![image](https://user-images.githubusercontent.com/43914109/144232994-14e0c276-a296-4347-8ae7-cff3a9c8cf31.png)
 
 Next, we define another helper function to display the annotation, with different colors indicating different labels:

```
colours = ['r', 'g', 'b']
def plot_image(img_tensor, annotation, display_ann = True):
    
    fig,ax = plt.subplots(1)
    img = img_tensor.cpu().data

    # Display the image
    ax.imshow(img.permute(1, 2, 0))
    n_boxes = len(annotation["boxes"])
    for i in range(n_boxes):
        box = annotation["boxes"][i]
        col = annotation["labels"][i]
        xmin, ymin, xmax, ymax = box

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor=colours[col-1],facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
        if display_ann:
          ax.annotate(class_names[col-1], ( xmin, ymin),color=colours[col-1], weight='bold', fontsize=10, ha='left', va='baseline' )

    plt.show()
```
Let's split the data into training set and test set with `random_split`. As we have 853 data points, we will use 700 of them for training and 153 for testing. We also define the data transformator and our dataloader

```
data_transform = transforms.Compose([transforms.ToTensor()])
def collate_fn(batch):
    return tuple(zip(*batch))

batch_size = 4
dataset = MaskDataset(data_transform)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [700, 153])
train_dl = torch.utils.data.DataLoader(
 train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle = True)
test_dl = torch.utils.data.DataLoader(
 test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle = True)
```
```
for imgs, annotations in test_dl: # take one batch for visualizing
      imgs = list(img.to(device) for img in imgs)
      annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
      break
for i in range(1):
  plot_image(imgs[i], annotations[i])
```
![image](https://user-images.githubusercontent.com/43914109/144233182-1e916106-c27f-4580-aedf-f81ed5c7c7ae.png)

## Define Model
We will employ  [Faster R-CNN](https://arxiv.org/abs/1506.01497) for our detector. Faster R-CNN is a model that predicts both bounding boxes and class scores for potential objects in the image. We will start from the model that was pre-trained on COCO train2017, and replace the last layer with our custom classifier. 


```
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    # i.e. we get the number of output features of the second last layer
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    # note that we need an extra class for the background, hence +1
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    return model

```
Now let's instantiate the model and the optimizer as well as the learning rate scheduler
```
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 3
model = get_instance_segmentation_model(num_classes).to(device)

# optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
```
And train the model for several epochs
```
num_epochs = 10
len_dataloader = len(train_dl)

for epoch in range(num_epochs):
    model.train()
    i = 0    
    epoch_loss = 0
    for imgs, annotations in train_dl:
        i += 1
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())        

        optimizer.zero_grad()
        losses.backward()
        optimizer.step() 
        
        print(f'Batch: {i}/{len_dataloader}, Loss: {losses.item()}')
        epoch_loss += losses.item()
    lr_scheduler.step()
    print(f'>>>>>>> Done epoch {epoch}, loss {epoch_loss/i}' )
```

In some case, the loaded model become too large and can take up a lot of memory. It is possible to delete unused variables and empty pytorch cache:

```
# avoid CUDA out of memory
import gc
gc.collect()
torch.cuda.empty_cache()

del losses, imgs, annotations,  
```

We can test the trained model on some unseen images:

```

batch_size = 8
test_dl = torch.utils.data.DataLoader(
 test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle = True)
for imgs, annotations in test_dl: # take one batch for testing
      imgs = list(img.to(device) for img in imgs)
      annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
      break
model.eval()

with torch.no_grad():
  preds = model(imgs[:batch_size])

for i in range(batch_size):
  print("Prediction")
  plot_image(imgs[i], preds[i], False)
  print("Target")
  plot_image(imgs[i], annotations[i], False)

```
We see that the model performance for the class 1 and 2 are worse than for the class 3 ("with_masks"). This can be due to the fact that these 2 classes are underpresented in the training data:


![42](https://user-images.githubusercontent.com/43914109/144233860-9c443518-15b9-40f3-ae80-121a00fd98d3.PNG)
![Capture](https://user-images.githubusercontent.com/43914109/144233871-660df07c-d3df-4ffc-9ce0-aacbf1cd0060.PNG)
![1](https://user-images.githubusercontent.com/43914109/144233878-abfae2b8-e987-4f34-970d-f5087cb8e067.PNG)
![2](https://user-images.githubusercontent.com/43914109/144233887-ceff242c-9f87-4113-abbf-80617fa4b578.PNG)
![3](https://user-images.githubusercontent.com/43914109/144233896-11b1873b-17fa-43ed-a1f1-c15582865c34.PNG)

## Save and Load model
Saving models for later uses and loading the saved models are as simple as follows:

```
# save model
torch.save(model.state_dict(),'model.pt')

# load model
model2 = get_instance_segmentation_model(3)
model2.load_state_dict(torch.load('model.pt'))
model2 = model2.to(device)
```
