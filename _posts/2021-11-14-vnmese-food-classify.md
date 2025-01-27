---
excerpt: Let us make a classificator for the wonderful Vietnamese dishes!
author_profile: true
title:  "Vietnamese Food Classificator"
categories:
  - machine-learning
tags:
  - deep learning
  - machine learning
  - computer vision
header:
  overlay_image: /assets/images/vnmesefood.jpg
  teaser: /assets/images/vnmesefood.jpg
  overlay_filter: 0.5
---
Six years away from home, what I miss the most about Vietnam are the wonderful dishes. I miss the freshness, the flavorfulness, the perfect balance within each dish. The Vietnamese cuisine is characterised with using fresh ingredients (most households do the grocery shopping everyday!), harmony of texture and flavors, minimal use of oil, and a lot of fresh herbs. 

In this post, we are going to employ the [fastai](https://www.fast.ai/) deep learning framework to train a classification model for Vietnamese dishes and serve the trained model as a web application with [Streamlit](https://streamlit.io/). For the classification task, we select a list of dishes we want to classify and download the images from the internet to create our custom dataset.  

A demo of the app is available [here](https://share.streamlit.io/qgiaong/vnmesefood_classsificator/main/app.py).

# Create and prepare the dataset
One of the most important step for any classification task is to create a labeled dataset, i.e., each data point is assigned to a specific class. For many tasks, there are already curated, well-maintained datasets that one can use, but it is not always the case. Instead, we can create a custom labeled image dataset by collecting data from the internet.

A simple way to search for training data is to use [Bing Image Downloader](https://github.com/gurugaurav/bing_image_downloader), which is a python package that helps download and search images from the Bing API. The package can be installed with `pip`. Note that the older version 1.0.4 yields much better result than the verion 1.1.1. Download the images for training is as simple as:


```python
from bing_image_downloader import downloader
#
for q in food_list:
   downloader.download(q, limit=200, output_dir="foods", adult_filter_off=True, force_replace=False, timeout=5)
```
where `food_list` contains the name of the food you want to classify. This will download the images in your local drive, with one folder for each name. Since the data are crawled from the web, we must make sure that the downloaded files are the actual images of the dished. Another postprocessing step is therefore needed. Check the downloaded images and delete the wrong ones if needed.

I have already prepared a dataset for our classification task. The zip file contains images of 20 Vietnamese dishes and can be downloaded from Google Drive:
```
!gdown  --id 1L8flvsHZ3lPftDWaqclAvhIb_vW1NwXc
```

As mentioned earlier, we are going to use the fastai framework for training. First of all, let's import the needed modules:
```python
from fastbook import *
from fastai.vision.widgets import *
from fastai.callback.fp16 import *
```
The path to the image folder can be specified as follows:
```python
path = Path('/vnmesefood')
fns = get_image_files(path)
failed = verify_images(fns) # check for invalid files
failed
```
Next, we define a DataBlock object for the training:

```python
vfoods = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
dls = vfoods.dataloaders(path)
dls.valid.show_batch(max_n=6, nrows=1)
```
![image](https://user-images.githubusercontent.com/43914109/147765691-cf5c41eb-5c98-46d1-823e-87ebdb37b8d5.png)

To mitigate overfitting, let's add some image augmentation to the dataset:
```python
vfoods = vfoods.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = vfoods.dataloaders(path, bs = 50)
dls.train.show_batch(max_n=4, nrows=1, unique=True)
```
![image](https://user-images.githubusercontent.com/43914109/147765708-3f0e0a2c-0d82-40b1-9075-811c16db7249.png)

# Define and train model
We employ the transfer learning technique to train our classificator. To be specific, we reuse the pretrained resnet18 architecture and finetune it on our training data. fastai provides a convenient way to find an appropriate learning rate for the training process:
```python
learn = cnn_learner(dls, resnet18, metrics=error_rate).to_fp16()
learn.lr_find()
```
![image](https://user-images.githubusercontent.com/43914109/147765725-ce04a0b7-a88f-4b31-9520-8ff14ddb8eca.png)

Finetuning can then be done in one line of code:
```python
learn.fine_tune(6, base_lr=0.00145, freeze_epochs=2)
```
![image](https://user-images.githubusercontent.com/43914109/147765758-a071fcc0-7607-49dc-8be6-c39540ceb1b9.png)

```python
learn.recorder.plot_loss()
```

# Visualize the training result
```python
learn.show_results(figsize=(20,10),  max_n=15)
```
![image](https://user-images.githubusercontent.com/43914109/147765776-4e1f60c7-c794-4196-903a-b442d9a5f820.png)

We can also plot the confusion matrix to see where the model often makes mistakes:
```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
```
![image](https://user-images.githubusercontent.com/43914109/147765785-be8ed4ec-a394-4eae-8550-ff909454b04d.png)

The confusion matrix might be hard to understand, we can call the `most_confused` function instead:

```python
interp.most_confused(min_val=5)
```
```
[('nem cuon', 'goi cuon', 9),
 ('goi cuon', 'nem cuon', 8),
 ('bun rieu', 'bun bo hue', 7)]
```
# Create a webapp for the classificator with Streamlit
[Streamlit](https://streamlit.io/) offers a fast way to create web apps for data science project. Let's create a simple web app so that one can upload an image and classify it with our trained model. 

First, we need to export the trained model 
```python
save_name = "vnmesefood_model"
learn.export(save_name)
```
To create a Streamlit app, create an "app.py" file and add the following requirements:
```python
import streamlit as st
from fastai.vision.all import *
from fastai.vision.widgets import *
import gc
import cv2
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
```
Then set some global configuration as follows:
```python
plt.style.use("seaborn")
# Enable garbage collection
gc.enable()

# Hide warnings
st.set_option("deprecation.showfileUploaderEncoding", False)
```
Next, let's create the web app title and banner:
```python
# page title
st.title('Vietnamese Food Classificator')
# Set the directory path
my_path = '.'

banner_path = my_path + '/banner.png'

# Read and display the banner
st.image(banner_path, use_column_width=True)

# App description
st.write(
    "**Get the name of the Vietnamese Food ! "
    "**")
st.markdown('***')
```
We also add an upload button for the user to upload an image:
```pythonpython
st.write("**Upload your Image**")
uploaded_image = st.file_uploader("Upload your image in JPG or PNG format", type=["jpg", "png"])
```

Next, we define a function to display some related plots:
```python
def plot_pred(img, learn_inf, k = 5):
    name,_, probs = learn_inf.predict(img)
    ids = np.argsort(-probs)[:k]
    top_names = learn_inf.dls.vocab[ids]
    top_probs = [round(p.numpy() * 100, 2) for p in probs[ids]]

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_axes([0,0,1,1])
    ax.bar(top_names,top_probs)
    return fig

```
To make a prediction, we just have to load the trained model as follows:
```python
def deploy(file_path=None, uploaded_image=uploaded_image, uploaded=True, demo=True):
    # Load the model and the weights
    learn_inf = load_learner('vnmesefood_model.pkl')
    st.markdown("image uploaded", unsafe_allow_html=True)
    st.image(uploaded_image, width=301, channels='BGR')

    # Display the uploaded/selected image
    st.markdown('***')
    st.markdown("Predicting...", unsafe_allow_html=True)
    img = PILImage.create(uploaded_image)
    pred, pred_idx, probs = learn_inf.predict(img)

    st.write(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
    fig = plot_pred(img, learn_inf)
    st.write(fig)

    gc.collect()
```
Finally, we just have to call the trained model everytime a new image is uploaded:

```python
if uploaded_image is not None:
    choice = 'Select an Image'
    # Deploy the model with the uploaded image
    deploy(uploaded_image, uploaded=True, demo=False)
    del uploaded_image
```
To start the web app, open the terminal and call
```
streamlit run app.py
```
![image](https://user-images.githubusercontent.com/43914109/147768429-3772d9b4-15e0-422e-b456-f1f3a5e749d9.png)

Voila! You have created a simple web app for food classification with fastai and Streamlit! The web app is still running locally, but you can also deploy it to a remote server, see [Deploy a Streamlit app](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app).




