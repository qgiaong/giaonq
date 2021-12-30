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
![image](https://user-images.githubusercontent.com/43914109/147786757-fbb83db1-5a6d-4f0a-9557-1593736b5d5b.png)
Next, we would like to understand the correlation between the target and other numerical features:
```python
corrmat = train_data.corr()
f, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corrmat, vmax=.8, square=True)
```
![image](https://user-images.githubusercontent.com/43914109/147786802-40ebc800-a6f8-444c-a0a3-e92b7ed3ffe1.png)
Next, we study the top 10 features that has the highest correlation to the target:
```
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
plt.subplots(figsize=(8, 6))
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
```
![image](https://user-images.githubusercontent.com/43914109/147786837-ad0ccb8c-a427-456d-9aba-23f0e5e16bd0.png)

Following points can be infered from the heatmap:


*  'OverallQual' (Overall material and finish quality), 'GrLivArea' (Above grade (ground) living area square feet) and 'TotalBsmtSF' (Total square feet of basement area) are strongly correlated with 'SalePrice', which totally makes sense
*   2 next highly collerated features are 'GarageCars' (Size of garage in car capacity) and 'GarageArea' (Size of garage in square feet). It is noteworthy that these two features indicate quite similar characteristics; indeed,  'GarageCars' is a consequence of 'GarageArea'. The same applies to 'TotalBsmtSF' (Total square feet of basement area) and '1stFloor' 

Next, we take a closer look to 5 intereseting features.
```
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_data[cols], size = 2.5)
plt.show();
```
![image](https://user-images.githubusercontent.com/43914109/147786881-f71edae1-f458-4948-becb-13aa98e8b1db.png)
Subsequently, we would like to study the data on missing data: 
```

```
