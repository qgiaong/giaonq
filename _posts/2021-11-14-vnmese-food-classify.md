---
author_profile: true
title:  "Vietnamese Food Classificator"
categories:
  - machine-learning
tags:
  - deep learning
  - machine learning
  - computer vision
header:
  image: /assets/images/bun_bo.jpg

---

# Create and prepare the dataset
We can create a custom dataset using the [Bing Image Downloader took](https://github.com/gurugaurav/bing_image_downloader). Note that the older version 1.0.4 yields much better result than the verion 1.1.1. Download the images for training is as simple as:

```
from bing_image_downloader import downloader

for q in food_list:
   downloader.download(q, limit=200, output_dir="foods", adult_filter_off=True, force_replace=False, timeout=5)
```
where food_list contains the name of the food you want to classify. This will download the images in your local drive, with one folder for each name. Since the data are crawled from the web, we must make sure that the downloaded files are the actual images of the dished. Delete the wrong images if needed.

I have already prepared a dataset for that. The zip file contains images of 20 Vietnamese dishes and can be downloaded from Google Drive:
```
!gdown  --id 1L8flvsHZ3lPftDWaqclAvhIb_vW1NwXc
```

In this post, we are going to use the fastai framework for training. First of all, let's import the needed modules:
```
from fastbook import *
from fastai.vision.widgets import *
from fastai.callback.fp16 import *
```
The path to the image folder can be specified as follows:
```
path = Path('/vnmesefood')
fns = get_image_files(path)
failed = verify_images(fns) # check for invalid files
failed
```
Next, we define a DataBlock object for the training:

```
vfoods = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
dls = vfoods.dataloaders(path)
dls.valid.show_batch(max_n=6, nrows=1)
```
To mitigate overfitting, let's add some image augmentation to the dataset:
```
vfoods = vfoods.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = vfoods.dataloaders(path, bs = 50)
dls.train.show_batch(max_n=4, nrows=1, unique=True)
```
# Define and train model
We employ the transfer learning technique to train our classificator. To be specific, we reuse the pretrained resnet18 architecture and finetune it on our training data. fastai provides a convenient way to find an appropriate learning rate for the training process:
```
learn = cnn_learner(dls, resnet18, metrics=error_rate).to_fp16()
learn.lr_find()
```
Finetuning can then be done in one line of code:
```
learn.fine_tune(6, base_lr=0.00145, freeze_epochs=2)
```
```
learn.recorder.plot_loss()
```

# Visualize the training result
```
learn.show_results(figsize=(20,10),  max_n=15)
```
We can also plot the confusion matrix to see where the model often makes mistakes:
```
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
```
The confusion matrix might be hard to understand, we can call the `most_confused` function instead:

```
interp.most_confused(min_val=5)
```
