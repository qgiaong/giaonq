---
excerpt: predict the house price given other indicators
author_profile: true
title:  "House Sale Price Prediction"
categories:
  - data science
tags:
  - regression
  - machine learning
  - data science
header:
  overlay_image: /assets/images/house_fund.jpg
  teaser: /assets/images/house_fund.jpg
  overlay_filter: 0.5
---
# Introduction
This blog post attempts to study the factors that influence the sale price of houses. To be specific, we will study the [Ames Housing dataset ](http://jse.amstat.org/v19n3/decock.pdf), which can be downloaded from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). 

# Data import and exploration
First, let's import the required packages:
```python
import datetime
import random
import os 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, scale
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
```
```python
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
```
```python
train_data.head()
```
