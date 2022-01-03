---
excerpt: set up a bot to notice stock indicators
author_profile: true
title:  "Create a Stack Bot for Stock Indexes Notification"
categories:
  - coding
tags:
  - web api
header:
  overlay_image: /assets/images/stock.jpg
  teaser: /assets/images/stock.jpg
  overlay_filter: 0.5
---
In this post, we are going to use the [Alpha Vantage API](https://www.alphavantage.co/) and the [Slack API](https://api.slack.com/) to create a Slack bot to notify us if the stock indicators meet some conditions of interests.
First of all, you need to sign up for the 2 APIs and get your private key for each.
```python
import requests
import json
import plotly as px

import csv
import requests
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

import os
import time
import re
from slackclient import SlackClient

import numpy as np
import argparse
```

# Get stock information from Alpha Vantage
Assume that we are interested in the following stock indexes: price, SMA, STOCH and MACDEXT. Next, we would like to get information about the "NVDA" stock.
The following functions define API calls to get such information:
```python
key = "YOUR_API_KEY"
def getPrice(symbol="NVDA"):
    CSV_URL = base_url  + "query?function=TIME_SERIES_DAILY&symbol=" + symbol
    CSV_URL += "&apikey=" + key
    
    with requests.Session() as s:
        res = s.get(CSV_URL).json()
    return res
```
```python
def getSMA(symbol="NVDA", interval="daily", slice="year1month1", time_period="14",series_type="open"):
    CSV_URL = base_url  + "query?function=SMA&symbol=" + symbol
    CSV_URL += "&interval=" + interval + "&time_period=" + time_period
    CSV_URL += "&series_type=" + series_type + "&apikey=" + key
    
    with requests.Session() as s:
        res = s.get(CSV_URL).json() 
    return res
 ```
 
 ```python
def getSTOCH(symbol="NVDA", interval="daily"):
  CSV_URL = base_url  + "query?function=STOCH&symbol=" + symbol
  CSV_URL += "&interval=" + interval + "&apikey=" + key
    
  with requests.Session() as s:
      res = s.get(CSV_URL).json()
  return res
```
```python
def getMACDEXT():
    CSV_URL = base_url  + "query?function=MACDEXT&symbol=" + "STM.DEX"
    CSV_URL += "&interval=" + "daily" +"&series_type="+"open"+"&apikey=" + key
    
    with requests.Session() as s:
        res = s.get(CSV_URL).json()
 
    return res
```
```python
price_list = getPrice()
sna_list = getSMA()
stoch_list = getSTOCH()
macd_list = getMACDEXT()
```
We can transform the raw data to Pandas dataframes for processing:
```python
prices_df = pd.DataFrame(price_list["Time Series (Daily)"]).transpose()[:n_days]
sma_df = pd.DataFrame(sna_list['Technical Analysis: SMA']).transpose()[:n_days]
stoch_df = pd.DataFrame(stoch_list['Technical Analysis: STOCH']).transpose()[:n_days]
macd_df = pd.DataFrame(macd_list['Technical Analysis: MACDEXT']).transpose()[:n_days]
```

We can also create a joint dataframe:
```python
join_df = prices_df.join(sma_df).join(stoch_df).join(macd_df)
join_df = join_df.rename_axis('date').reset_index()

join_df = join_df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low','4. close': 'close','5. volume': 'volume'})

columns = ["open", "high","low","close","volume","SMA", "SlowK","SlowD", "MACD_Hist", "MACD", "MACD_Signal"]
for c in columns:
  join_df[c] = pd.to_numeric(join_df[c], downcast="float")
join_df.head()
```
```python
join_df.info()
```
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100 entries, 0 to 99
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   date         100 non-null    object 
     1   open         100 non-null    float32
     2   high         100 non-null    float32
     3   low          100 non-null    float32
     4   close        100 non-null    float32
     5   volume       100 non-null    float32
     6   SMA          99 non-null     float32
     7   SlowD        99 non-null     float32
     8   SlowK        99 non-null     float32
     9   MACD         98 non-null     float32
     10  MACD_Hist    98 non-null     float32
     11  MACD_Signal  98 non-null     float32
    dtypes: float32(11), object(1)
    memory usage: 5.2+ KB
    
 Plotly provides a convenient way to plot time series data:
```python
df_sma = pd.melt(join_df, id_vars = ['date'], value_vars = ['SMA', 'close'])
xi = x for (x, y) in intersections]
yi = [y for (x, y) in intersections]
fig1 = px.line(df_sma, x = 'date', y = 'value', color = 'variable')
# Show plot
fig1.show()
```

{% include_relative stocks/sma_close.html %}

The resulting plot can be saved as interactive (.html) or static (e.g. .png) images:
```python
fig1.write_image("sma_close.webp")
fig1.write_html("sma_close.html")
```
 
 
# Create a Slack Bot for notification
You will need the [slack_sdk](https://pypi.org/project/slack-sdk/) package to make the API calls directly with python:
```python
import logging
import os
# Import WebClient from Python SDK (github.com/slackapi/python-slack-sdk)
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
```
```python
def send_message(msg, channel="money"):
  client = WebClient(token=SLACK_BOT_TOKEN)
  logger = logging.getLogger(__name__)
  try:
    # Call the conversations.list method using the WebClient
    result = client.chat_postMessage(
        channel=channel,
        text=msg,  # You could also use a blocks[] array to send richer content
        username='Cute Bot',
        icon_emoji=':robot_face:'
    )
    logger.info(result)
  except SlackApiError as e:
      print(f"Error: {e}")
```
```python
def send_file(file_name, init_cmt = "Here's my file :smile:",  channel="money"):
  client = WebClient(token=SLACK_BOT_TOKEN)
  logger = logging.getLogger(__name__)
  try:
    # Call the conversations.list method using the WebClient
    result = client.files_upload(
        channels=channel,
        initial_comment=init_cmt,
        file=file_name,
        username='Cute Bot',
        icon_emoji=':robot_face:'
    )
    # Log the result
    logger.info(result)
  except SlackApiError as e:
      print(f"Error: {e}")
```
We can try sending something to our channel:
```python
send_message("hello")
send_file("sma_close.webp")
```

# Deploy the Stock Bot on server (coming ....)
Now as we are all set, we can deploy our bot to a server so that it checks the price every 5 minutes and sends you a Slack notice if some predefined conditions are met. We can deploy to some remote server such as Heroku, or set up a local server on Raspberry Pi and deploy to it. I will come back to this in a later post.
