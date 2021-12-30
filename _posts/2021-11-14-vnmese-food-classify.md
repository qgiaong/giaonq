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




