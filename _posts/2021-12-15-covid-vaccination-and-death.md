---
excerpt: 
author_profile: true
title:  "Covid Deaths and Covid Vaccination"
categories:
  - data science
tags:
  - regression
  - machine learning
  - data science
header:
  overlay_image: /assets/images/covid.jpg
  teaser: /assets/images/covid.jpg
  overlay_filter: 0.5
---


<a href="https://colab.research.google.com/github/qgiaong/blogs/blob/main/CovidVaccinationAndDeath.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


# Covid Vaccination



```python
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from   plotly.subplots import make_subplots
from   plotly import subplots
```


```python
df = pd.read_csv("train/vaccinations.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>iso_code</th>
      <th>date</th>
      <th>total_vaccinations</th>
      <th>people_vaccinated</th>
      <th>people_fully_vaccinated</th>
      <th>total_boosters</th>
      <th>daily_vaccinations_raw</th>
      <th>daily_vaccinations</th>
      <th>total_vaccinations_per_hundred</th>
      <th>people_vaccinated_per_hundred</th>
      <th>people_fully_vaccinated_per_hundred</th>
      <th>total_boosters_per_hundred</th>
      <th>daily_vaccinations_per_million</th>
      <th>daily_people_vaccinated</th>
      <th>daily_people_vaccinated_per_hundred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2021-02-22</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2021-02-23</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1367.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>34.0</td>
      <td>1367.0</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2021-02-24</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1367.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>34.0</td>
      <td>1367.0</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2021-02-25</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1367.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>34.0</td>
      <td>1367.0</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2021-02-26</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1367.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>34.0</td>
      <td>1367.0</td>
      <td>0.003</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 62102 entries, 0 to 62101
    Data columns (total 16 columns):
     #   Column                               Non-Null Count  Dtype  
    ---  ------                               --------------  -----  
     0   location                             62102 non-null  object 
     1   iso_code                             62102 non-null  object 
     2   date                                 62102 non-null  object 
     3   total_vaccinations                   35172 non-null  float64
     4   people_vaccinated                    33595 non-null  float64
     5   people_fully_vaccinated              30617 non-null  float64
     6   total_boosters                       6291 non-null   float64
     7   daily_vaccinations_raw               29452 non-null  float64
     8   daily_vaccinations                   61784 non-null  float64
     9   total_vaccinations_per_hundred       35172 non-null  float64
     10  people_vaccinated_per_hundred        33595 non-null  float64
     11  people_fully_vaccinated_per_hundred  30617 non-null  float64
     12  total_boosters_per_hundred           6291 non-null   float64
     13  daily_vaccinations_per_million       61784 non-null  float64
     14  daily_people_vaccinated              60495 non-null  float64
     15  daily_people_vaccinated_per_hundred  60495 non-null  float64
    dtypes: float64(13), object(3)
    memory usage: 7.6+ MB
    


```python
df['location'].value_counts()
```




    World               351
    High income         351
    Europe              351
    European Union      351
    Denmark             350
                       ... 
    Pitcairn             85
    Tanzania             83
    Falkland Islands     67
    Niue                 43
    Burundi              25
    Name: location, Length: 235, dtype: int64




```python
sorted_df = df.groupby('location').max().sort_values('total_vaccinations', ascending=False).dropna(subset=['total_vaccinations'])
sorted_df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>iso_code</th>
      <th>date</th>
      <th>total_vaccinations</th>
      <th>people_vaccinated</th>
      <th>people_fully_vaccinated</th>
      <th>total_boosters</th>
      <th>daily_vaccinations_raw</th>
      <th>daily_vaccinations</th>
      <th>total_vaccinations_per_hundred</th>
      <th>people_vaccinated_per_hundred</th>
      <th>people_fully_vaccinated_per_hundred</th>
      <th>total_boosters_per_hundred</th>
      <th>daily_vaccinations_per_million</th>
      <th>daily_people_vaccinated</th>
      <th>daily_people_vaccinated_per_hundred</th>
    </tr>
    <tr>
      <th>location</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>World</th>
      <td>OWID_WRL</td>
      <td>2021-11-16</td>
      <td>7.558708e+09</td>
      <td>4.120535e+09</td>
      <td>3.234759e+09</td>
      <td>173232566.0</td>
      <td>56343781.0</td>
      <td>43233330.0</td>
      <td>95.98</td>
      <td>52.32</td>
      <td>41.08</td>
      <td>2.20</td>
      <td>5490.0</td>
      <td>100631920.0</td>
      <td>1.278</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>OWID_ASI</td>
      <td>2021-11-16</td>
      <td>5.117174e+09</td>
      <td>2.820122e+09</td>
      <td>2.137725e+09</td>
      <td>78462762.0</td>
      <td>43192331.0</td>
      <td>33335559.0</td>
      <td>109.38</td>
      <td>60.28</td>
      <td>45.69</td>
      <td>1.68</td>
      <td>7125.0</td>
      <td>95197815.0</td>
      <td>2.035</td>
    </tr>
    <tr>
      <th>Upper middle income</th>
      <td>OWID_UMC</td>
      <td>2021-11-16</td>
      <td>3.595079e+09</td>
      <td>1.836460e+09</td>
      <td>1.603370e+09</td>
      <td>85345208.0</td>
      <td>30996692.0</td>
      <td>27439068.0</td>
      <td>143.02</td>
      <td>73.06</td>
      <td>63.79</td>
      <td>3.40</td>
      <td>10916.0</td>
      <td>92139369.0</td>
      <td>3.666</td>
    </tr>
    <tr>
      <th>China</th>
      <td>CHN</td>
      <td>2021-11-15</td>
      <td>2.396045e+09</td>
      <td>1.185237e+09</td>
      <td>1.073845e+09</td>
      <td>49440000.0</td>
      <td>24741000.0</td>
      <td>22424286.0</td>
      <td>165.91</td>
      <td>82.07</td>
      <td>74.35</td>
      <td>3.42</td>
      <td>15527.0</td>
      <td>5850649.0</td>
      <td>0.405</td>
    </tr>
    <tr>
      <th>Lower middle income</th>
      <td>OWID_LMC</td>
      <td>2021-11-16</td>
      <td>2.170151e+09</td>
      <td>1.367115e+09</td>
      <td>8.051733e+08</td>
      <td>3703110.0</td>
      <td>33454856.0</td>
      <td>16674499.0</td>
      <td>65.16</td>
      <td>41.05</td>
      <td>24.17</td>
      <td>0.11</td>
      <td>5006.0</td>
      <td>10636385.0</td>
      <td>0.319</td>
    </tr>
    <tr>
      <th>High income</th>
      <td>OWID_HIC</td>
      <td>2021-11-16</td>
      <td>1.749729e+09</td>
      <td>8.856635e+08</td>
      <td>8.088929e+08</td>
      <td>84184248.0</td>
      <td>11915046.0</td>
      <td>8396718.0</td>
      <td>144.02</td>
      <td>72.90</td>
      <td>66.58</td>
      <td>6.93</td>
      <td>6911.0</td>
      <td>5570179.0</td>
      <td>0.458</td>
    </tr>
    <tr>
      <th>India</th>
      <td>IND</td>
      <td>2021-11-16</td>
      <td>1.133688e+09</td>
      <td>7.560527e+08</td>
      <td>3.776355e+08</td>
      <td>NaN</td>
      <td>18627269.0</td>
      <td>10037995.0</td>
      <td>81.36</td>
      <td>54.26</td>
      <td>27.10</td>
      <td>NaN</td>
      <td>7204.0</td>
      <td>6785334.0</td>
      <td>0.487</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>OWID_EUR</td>
      <td>2021-11-16</td>
      <td>9.001484e+08</td>
      <td>4.586002e+08</td>
      <td>4.232705e+08</td>
      <td>38536654.0</td>
      <td>6311580.0</td>
      <td>5128558.0</td>
      <td>120.19</td>
      <td>61.23</td>
      <td>56.51</td>
      <td>5.15</td>
      <td>6848.0</td>
      <td>2785818.0</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>North America</th>
      <td>OWID_NAM</td>
      <td>2021-11-16</td>
      <td>7.185189e+08</td>
      <td>3.747111e+08</td>
      <td>3.197385e+08</td>
      <td>33552453.0</td>
      <td>8168891.0</td>
      <td>4172759.0</td>
      <td>120.44</td>
      <td>62.81</td>
      <td>53.60</td>
      <td>5.62</td>
      <td>6994.0</td>
      <td>2556767.0</td>
      <td>0.429</td>
    </tr>
    <tr>
      <th>European Union</th>
      <td>OWID_EUN</td>
      <td>2021-11-16</td>
      <td>6.108846e+08</td>
      <td>3.118220e+08</td>
      <td>2.966872e+08</td>
      <td>21480999.0</td>
      <td>5193009.0</td>
      <td>4075710.0</td>
      <td>136.61</td>
      <td>69.73</td>
      <td>66.34</td>
      <td>4.80</td>
      <td>9114.0</td>
      <td>2352258.0</td>
      <td>0.526</td>
    </tr>
    <tr>
      <th>South America</th>
      <td>OWID_SAM</td>
      <td>2021-11-16</td>
      <td>5.580190e+08</td>
      <td>3.072378e+08</td>
      <td>2.399412e+08</td>
      <td>22115591.0</td>
      <td>12998583.0</td>
      <td>3976259.0</td>
      <td>128.50</td>
      <td>70.75</td>
      <td>55.25</td>
      <td>5.09</td>
      <td>9156.0</td>
      <td>2549891.0</td>
      <td>0.587</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>USA</td>
      <td>2021-11-16</td>
      <td>4.433742e+08</td>
      <td>2.276919e+08</td>
      <td>1.939638e+08</td>
      <td>30651760.0</td>
      <td>4516889.0</td>
      <td>3498728.0</td>
      <td>131.83</td>
      <td>67.70</td>
      <td>57.67</td>
      <td>9.11</td>
      <td>10403.0</td>
      <td>2028734.0</td>
      <td>0.603</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>BRA</td>
      <td>2021-11-16</td>
      <td>2.971040e+08</td>
      <td>1.623420e+08</td>
      <td>1.279980e+08</td>
      <td>11814702.0</td>
      <td>11231782.0</td>
      <td>2595170.0</td>
      <td>138.84</td>
      <td>75.86</td>
      <td>59.81</td>
      <td>5.52</td>
      <td>12127.0</td>
      <td>1394879.0</td>
      <td>0.652</td>
    </tr>
    <tr>
      <th>Africa</th>
      <td>OWID_AFR</td>
      <td>2021-11-16</td>
      <td>2.167046e+08</td>
      <td>1.347350e+08</td>
      <td>9.135311e+07</td>
      <td>280269.0</td>
      <td>5687163.0</td>
      <td>2030907.0</td>
      <td>15.78</td>
      <td>9.81</td>
      <td>6.65</td>
      <td>0.02</td>
      <td>1479.0</td>
      <td>1285011.0</td>
      <td>0.094</td>
    </tr>
    <tr>
      <th>Indonesia</th>
      <td>IDN</td>
      <td>2021-11-16</td>
      <td>2.166636e+08</td>
      <td>1.312929e+08</td>
      <td>8.537068e+07</td>
      <td>NaN</td>
      <td>3087420.0</td>
      <td>1901294.0</td>
      <td>78.40</td>
      <td>47.51</td>
      <td>30.89</td>
      <td>NaN</td>
      <td>6880.0</td>
      <td>1160342.0</td>
      <td>0.420</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>JPN</td>
      <td>2021-11-16</td>
      <td>1.951119e+08</td>
      <td>9.935584e+07</td>
      <td>9.575607e+07</td>
      <td>NaN</td>
      <td>6586453.0</td>
      <td>1997542.0</td>
      <td>154.79</td>
      <td>78.82</td>
      <td>75.97</td>
      <td>NaN</td>
      <td>15847.0</td>
      <td>1156833.0</td>
      <td>0.918</td>
    </tr>
    <tr>
      <th>Mexico</th>
      <td>MEX</td>
      <td>2021-11-16</td>
      <td>1.298744e+08</td>
      <td>7.545903e+07</td>
      <td>6.340724e+07</td>
      <td>NaN</td>
      <td>7246123.0</td>
      <td>1648223.0</td>
      <td>99.70</td>
      <td>57.93</td>
      <td>48.68</td>
      <td>NaN</td>
      <td>12653.0</td>
      <td>762995.0</td>
      <td>0.586</td>
    </tr>
    <tr>
      <th>Pakistan</th>
      <td>PAK</td>
      <td>2021-11-16</td>
      <td>1.197385e+08</td>
      <td>7.853426e+07</td>
      <td>4.860066e+07</td>
      <td>NaN</td>
      <td>1703092.0</td>
      <td>1280906.0</td>
      <td>53.17</td>
      <td>34.87</td>
      <td>21.58</td>
      <td>NaN</td>
      <td>5688.0</td>
      <td>921954.0</td>
      <td>0.409</td>
    </tr>
    <tr>
      <th>Turkey</th>
      <td>TUR</td>
      <td>2021-11-16</td>
      <td>1.187278e+08</td>
      <td>5.590454e+07</td>
      <td>4.977780e+07</td>
      <td>13045462.0</td>
      <td>1796891.0</td>
      <td>1264431.0</td>
      <td>139.61</td>
      <td>65.74</td>
      <td>58.53</td>
      <td>15.34</td>
      <td>14868.0</td>
      <td>1155560.0</td>
      <td>1.359</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>DEU</td>
      <td>2021-11-16</td>
      <td>1.156567e+08</td>
      <td>5.836668e+07</td>
      <td>5.628233e+07</td>
      <td>4368783.0</td>
      <td>1428605.0</td>
      <td>875110.0</td>
      <td>137.85</td>
      <td>69.57</td>
      <td>67.08</td>
      <td>5.21</td>
      <td>10430.0</td>
      <td>592809.0</td>
      <td>0.707</td>
    </tr>
  </tbody>
</table>
</div>




```python
# drop aggregate rows
sorted_df = sorted_df[~sorted_df['iso_code'].astype(str).str.startswith('OWID')]
```


```python
sorted_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>iso_code</th>
      <th>date</th>
      <th>total_vaccinations</th>
      <th>people_vaccinated</th>
      <th>people_fully_vaccinated</th>
      <th>total_boosters</th>
      <th>daily_vaccinations_raw</th>
      <th>daily_vaccinations</th>
      <th>total_vaccinations_per_hundred</th>
      <th>people_vaccinated_per_hundred</th>
      <th>people_fully_vaccinated_per_hundred</th>
      <th>total_boosters_per_hundred</th>
      <th>daily_vaccinations_per_million</th>
      <th>daily_people_vaccinated</th>
      <th>daily_people_vaccinated_per_hundred</th>
    </tr>
    <tr>
      <th>location</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>China</th>
      <td>CHN</td>
      <td>2021-11-15</td>
      <td>2.396045e+09</td>
      <td>1.185237e+09</td>
      <td>1.073845e+09</td>
      <td>49440000.0</td>
      <td>24741000.0</td>
      <td>22424286.0</td>
      <td>165.91</td>
      <td>82.07</td>
      <td>74.35</td>
      <td>3.42</td>
      <td>15527.0</td>
      <td>5850649.0</td>
      <td>0.405</td>
    </tr>
    <tr>
      <th>India</th>
      <td>IND</td>
      <td>2021-11-16</td>
      <td>1.133688e+09</td>
      <td>7.560527e+08</td>
      <td>3.776355e+08</td>
      <td>NaN</td>
      <td>18627269.0</td>
      <td>10037995.0</td>
      <td>81.36</td>
      <td>54.26</td>
      <td>27.10</td>
      <td>NaN</td>
      <td>7204.0</td>
      <td>6785334.0</td>
      <td>0.487</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>USA</td>
      <td>2021-11-16</td>
      <td>4.433742e+08</td>
      <td>2.276919e+08</td>
      <td>1.939638e+08</td>
      <td>30651760.0</td>
      <td>4516889.0</td>
      <td>3498728.0</td>
      <td>131.83</td>
      <td>67.70</td>
      <td>57.67</td>
      <td>9.11</td>
      <td>10403.0</td>
      <td>2028734.0</td>
      <td>0.603</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>BRA</td>
      <td>2021-11-16</td>
      <td>2.971040e+08</td>
      <td>1.623420e+08</td>
      <td>1.279980e+08</td>
      <td>11814702.0</td>
      <td>11231782.0</td>
      <td>2595170.0</td>
      <td>138.84</td>
      <td>75.86</td>
      <td>59.81</td>
      <td>5.52</td>
      <td>12127.0</td>
      <td>1394879.0</td>
      <td>0.652</td>
    </tr>
    <tr>
      <th>Indonesia</th>
      <td>IDN</td>
      <td>2021-11-16</td>
      <td>2.166636e+08</td>
      <td>1.312929e+08</td>
      <td>8.537068e+07</td>
      <td>NaN</td>
      <td>3087420.0</td>
      <td>1901294.0</td>
      <td>78.40</td>
      <td>47.51</td>
      <td>30.89</td>
      <td>NaN</td>
      <td>6880.0</td>
      <td>1160342.0</td>
      <td>0.420</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12, 6))
sns.barplot(data=sorted_df[:10],x=sorted_df.index[:10],y='people_fully_vaccinated')
plt.title('Top 10 Nations with highest number of people fully vaccinated against COVID-19')
plt.ylabel('Number of people fully vaccinated')
plt.xlabel('Countries')
```




    Text(0.5, 0, 'Countries')




    
![png](/_posts/covid/output_13_1.png)
    



```python
sorted_df['people_not_fully_vaccinated_per_hundred'] =  100-sorted_df['people_fully_vaccinated_per_hundred']
# estimate population
sorted_df['population'] = sorted_df['people_fully_vaccinated']/sorted_df['people_fully_vaccinated_per_hundred'] 
sorted_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>iso_code</th>
      <th>date</th>
      <th>total_vaccinations</th>
      <th>people_vaccinated</th>
      <th>people_fully_vaccinated</th>
      <th>total_boosters</th>
      <th>daily_vaccinations_raw</th>
      <th>daily_vaccinations</th>
      <th>total_vaccinations_per_hundred</th>
      <th>people_vaccinated_per_hundred</th>
      <th>people_fully_vaccinated_per_hundred</th>
      <th>total_boosters_per_hundred</th>
      <th>daily_vaccinations_per_million</th>
      <th>daily_people_vaccinated</th>
      <th>daily_people_vaccinated_per_hundred</th>
      <th>people_not_fully_vaccinated_per_hundred</th>
      <th>population</th>
    </tr>
    <tr>
      <th>location</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>China</th>
      <td>CHN</td>
      <td>2021-11-15</td>
      <td>2.396045e+09</td>
      <td>1.185237e+09</td>
      <td>1.073845e+09</td>
      <td>49440000.0</td>
      <td>24741000.0</td>
      <td>22424286.0</td>
      <td>165.91</td>
      <td>82.07</td>
      <td>74.35</td>
      <td>3.42</td>
      <td>15527.0</td>
      <td>5850649.0</td>
      <td>0.405</td>
      <td>25.65</td>
      <td>1.444311e+07</td>
    </tr>
    <tr>
      <th>India</th>
      <td>IND</td>
      <td>2021-11-16</td>
      <td>1.133688e+09</td>
      <td>7.560527e+08</td>
      <td>3.776355e+08</td>
      <td>NaN</td>
      <td>18627269.0</td>
      <td>10037995.0</td>
      <td>81.36</td>
      <td>54.26</td>
      <td>27.10</td>
      <td>NaN</td>
      <td>7204.0</td>
      <td>6785334.0</td>
      <td>0.487</td>
      <td>72.90</td>
      <td>1.393489e+07</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>USA</td>
      <td>2021-11-16</td>
      <td>4.433742e+08</td>
      <td>2.276919e+08</td>
      <td>1.939638e+08</td>
      <td>30651760.0</td>
      <td>4516889.0</td>
      <td>3498728.0</td>
      <td>131.83</td>
      <td>67.70</td>
      <td>57.67</td>
      <td>9.11</td>
      <td>10403.0</td>
      <td>2028734.0</td>
      <td>0.603</td>
      <td>42.33</td>
      <td>3.363340e+06</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>BRA</td>
      <td>2021-11-16</td>
      <td>2.971040e+08</td>
      <td>1.623420e+08</td>
      <td>1.279980e+08</td>
      <td>11814702.0</td>
      <td>11231782.0</td>
      <td>2595170.0</td>
      <td>138.84</td>
      <td>75.86</td>
      <td>59.81</td>
      <td>5.52</td>
      <td>12127.0</td>
      <td>1394879.0</td>
      <td>0.652</td>
      <td>40.19</td>
      <td>2.140078e+06</td>
    </tr>
    <tr>
      <th>Indonesia</th>
      <td>IDN</td>
      <td>2021-11-16</td>
      <td>2.166636e+08</td>
      <td>1.312929e+08</td>
      <td>8.537068e+07</td>
      <td>NaN</td>
      <td>3087420.0</td>
      <td>1901294.0</td>
      <td>78.40</td>
      <td>47.51</td>
      <td>30.89</td>
      <td>NaN</td>
      <td>6880.0</td>
      <td>1160342.0</td>
      <td>0.420</td>
      <td>69.11</td>
      <td>2.763700e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_df = sorted_df[['people_fully_vaccinated_per_hundred', 'people_not_fully_vaccinated_per_hundred','population']]
plot_df['location'] = plot_df.index
plot_df = plot_df.sort_values('population', ascending=False)
plot_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>people_fully_vaccinated_per_hundred</th>
      <th>people_not_fully_vaccinated_per_hundred</th>
      <th>population</th>
      <th>location</th>
    </tr>
    <tr>
      <th>location</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Burundi</th>
      <td>0.00</td>
      <td>100.00</td>
      <td>inf</td>
      <td>Burundi</td>
    </tr>
    <tr>
      <th>China</th>
      <td>74.35</td>
      <td>25.65</td>
      <td>1.444311e+07</td>
      <td>China</td>
    </tr>
    <tr>
      <th>India</th>
      <td>27.10</td>
      <td>72.90</td>
      <td>1.393489e+07</td>
      <td>India</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>57.67</td>
      <td>42.33</td>
      <td>3.363340e+06</td>
      <td>United States</td>
    </tr>
    <tr>
      <th>Indonesia</th>
      <td>30.89</td>
      <td>69.11</td>
      <td>2.763700e+06</td>
      <td>Indonesia</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Montserrat</th>
      <td>28.45</td>
      <td>71.55</td>
      <td>4.980668e+01</td>
      <td>Montserrat</td>
    </tr>
    <tr>
      <th>Falkland Islands</th>
      <td>50.31</td>
      <td>49.69</td>
      <td>3.528126e+01</td>
      <td>Falkland Islands</td>
    </tr>
    <tr>
      <th>Niue</th>
      <td>71.25</td>
      <td>28.75</td>
      <td>1.614035e+01</td>
      <td>Niue</td>
    </tr>
    <tr>
      <th>Tokelau</th>
      <td>70.76</td>
      <td>29.24</td>
      <td>1.368005e+01</td>
      <td>Tokelau</td>
    </tr>
    <tr>
      <th>Pitcairn</th>
      <td>100.00</td>
      <td>0.00</td>
      <td>4.700000e-01</td>
      <td>Pitcairn</td>
    </tr>
  </tbody>
</table>
<p>217 rows × 4 columns</p>
</div>




```python
plot_df = plot_df[1:11].drop('population', 1) # drop first row
```


```python
ax = plot_df.plot(figsize = (12, 6),
    x = 'location',
    kind = 'barh',
    stacked = True,
    title = 'Percentage of People Fully Vaccinated of top 10 most populous countries ',
    mark_right = True,
    colormap='Paired')

ax.set_xlabel("Percentage")
ax.set_ylabel("Country")
```




    Text(0, 0.5, 'Country')




    
![png](/_posts/covid/output_17_1.png)
    



```python
# Covid deaths over the time period
fig = px.choropleth(data_frame=sorted_df, locations='iso_code',
                   color='people_fully_vaccinated_per_hundred')
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="1ec06650-dacc-438b-bc31-5866e34c02a6" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("1ec06650-dacc-438b-bc31-5866e34c02a6")) {
                    Plotly.newPlot(
                        '1ec06650-dacc-438b-bc31-5866e34c02a6',
                        [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "iso_code=%{location}<br>people_fully_vaccinated_per_hundred=%{z}", "locations": ["CHN", "IND", "USA", "BRA", "IDN", "JPN", "MEX", "PAK", "TUR", "DEU", "RUS", "GBR", "FRA", "VNM", "IRN", "ITA", "THA", "BGD", "KOR", "ESP", "PHL", "ARG", "CAN", "COL", "MYS", "MAR", "SAU", "POL", "CHL", "AUS", "PER", "EGY", "UZB", "LKA", "KHM", "TWN", "CUB", "NLD", "ZAF", "VEN", "ECU", "MMR", "ARE", "UKR", "BEL", "KAZ", "PRT", "NPL", "ISR", "SWE", "ROU", "DOM", "GRC", "HUN", "CZE", "AUT", "DZA", "CHE", "IRQ", "AZE", "SGP", "TUN", "GTM", "HKG", "NGA", "SLV", "DNK", "FIN", "BOL", "NOR", "AGO", "JOR", "HND", "TKM", "SRB", "MOZ", "IRL", "NZL", "RWA", "CRI", "URY", "ZWE", "KEN", "OMN", "PRY", "PAN", "BLR", "ETH", "QAT", "SVK", "TJK", "UGA", "MNG", "LAO", "AFG", "HRV", "CIV", "LTU", "LBN", "GHA", "BGR", "PSE", "BHR", "KWT", "GIN", "SVN", "LBY", "LVA", "GEO", "ALB", "SEN", "KGZ", "MUS", "MRT", "NIC", "SDN", "MKD", "MDA", "BIH", "EST", "TGO", "SYR", "MWI", "TTO", "CYP", "FJI", "BTN", "BWA", "JAM", "TZA", "ZMB", "TLS", "SOM", "NER", "MLT", "LUX", "ARM", "MAC", "MDV", "YEM", "SLE", "BRN", "BFA", "GUY", "NAM", "MLI", "ISL", "MDG", "COG", "CPV", "CMR", "MNE", "COM", "SUR", "GNQ", "LBR", "CAF", "BLZ", "LSO", "BEN", "NCL", "GNB", "PYF", "SWZ", "PNG", "BRB", "GMB", "BHS", "GAB", "TCD", "WSM", "CUW", "JEY", "SYC", "SLB", "COD", "ABW", "HTI", "SSD", "IMN", "VUT", "CYM", "ATG", "STP", "AND", "GGY", "TON", "BMU", "LCA", "GIB", "DJI", "FRO", "GRL", "KIR", "GRD", "TCA", "DMA", "SXM", "KNA", "MCO", "VCT", "LIE", "SMR", "BES", "VGB", "COK", "AIA", "NRU", "TUV", "WLF", "SHN", "FLK", "MSR", "NIU", "TKL", "BDI", "PCN"], "name": "", "type": "choropleth", "z": [74.35, 27.1, 57.67, 59.81, 30.89, 75.97, 48.68, 21.58, 58.53, 67.08, 35.45, 67.51, 68.74, 36.77, 49.61, 72.69, 53.06, 19.93, 78.44, 80.18, 35.54, 60.26, 75.47, 44.75, 75.85, 60.23, 62.46, 53.34, 81.89, 69.96, 49.41, 12.73, 17.33, 63.61, 78.04, 42.88, 76.69, 73.2, 22.62, 32.3, 58.51, 15.05, 88.4, 20.91, 74.15, 41.28, 87.56, 25.06, 61.97, 68.68, 36.06, 50.04, 62.49, 60.0, 57.9, 63.72, 11.05, 64.62, 9.77, 43.92, 91.91, 40.3, 19.97, 59.35, 1.51, 61.32, 76.22, 71.52, 33.37, 69.1, 5.91, 35.44, 36.0, 52.41, 44.42, 8.09, 75.62, 67.2, 18.98, 58.97, 75.64, 17.98, 3.96, 52.32, 34.74, 54.77, 24.34, 1.18, 75.7, 42.62, 21.92, 1.94, 64.45, 37.63, 8.0, 45.89, 4.14, 65.22, 23.92, 2.63, 23.89, 25.3, 65.75, 21.33, 5.75, 54.52, 7.82, 60.23, 24.41, 32.17, 5.11, 12.54, 67.39, 13.89, 8.4, 1.3, 37.74, 22.54, 22.08, 58.61, 5.69, 3.95, 2.96, 44.85, 64.22, 63.83, 71.78, 14.9, 16.19, 1.57, 3.26, 28.44, 3.45, 1.82, 83.48, 65.91, 8.51, 54.97, 66.44, 1.13, 3.72, 69.15, 1.38, 33.33, 10.61, 1.3, 81.48, 0.65, 2.22, 41.17, 0.65, 39.75, 22.31, 35.77, 13.43, 7.17, 6.54, 46.35, 15.72, 2.13, 57.11, 0.89, 54.32, 21.29, 1.17, 46.15, 8.95, 32.31, 4.75, 0.39, 43.09, 56.89, 74.02, 78.24, 5.05, 0.05, 72.28, 0.42, 0.58, 75.78, 11.73, 82.95, 53.37, 12.79, 64.04, 21.48, 36.82, 76.28, 23.99, 118.18, 2.6, 77.76, 65.52, 12.51, 29.22, 69.17, 35.68, 56.46, 46.05, 58.98, 18.76, 64.8, 65.65, 63.29, 54.14, 66.73, 60.55, 66.69, 49.34, 52.91, 57.93, 50.31, 28.45, 71.25, 70.76, 0.0, 100.0]}],
                        {"coloraxis": {"colorbar": {"title": {"text": "people_fully_vaccinated_per_hundred"}}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "geo": {"center": {}, "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]}}, "legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('1ec06650-dacc-438b-bc31-5866e34c02a6');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


## Vaccination in Germany over time


```python
df_de = df[df['iso_code'] == 'DEU'].sort_values('date')
df_de.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>iso_code</th>
      <th>date</th>
      <th>total_vaccinations</th>
      <th>people_vaccinated</th>
      <th>people_fully_vaccinated</th>
      <th>total_boosters</th>
      <th>daily_vaccinations_raw</th>
      <th>daily_vaccinations</th>
      <th>total_vaccinations_per_hundred</th>
      <th>people_vaccinated_per_hundred</th>
      <th>people_fully_vaccinated_per_hundred</th>
      <th>total_boosters_per_hundred</th>
      <th>daily_vaccinations_per_million</th>
      <th>daily_people_vaccinated</th>
      <th>daily_people_vaccinated_per_hundred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20817</th>
      <td>Germany</td>
      <td>DEU</td>
      <td>2020-12-27</td>
      <td>24355.0</td>
      <td>24344.0</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20818</th>
      <td>Germany</td>
      <td>DEU</td>
      <td>2020-12-28</td>
      <td>42459.0</td>
      <td>42384.0</td>
      <td>75.0</td>
      <td>NaN</td>
      <td>18104.0</td>
      <td>18104.0</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>216.0</td>
      <td>18040.0</td>
      <td>0.022</td>
    </tr>
    <tr>
      <th>20819</th>
      <td>Germany</td>
      <td>DEU</td>
      <td>2020-12-29</td>
      <td>93182.0</td>
      <td>92454.0</td>
      <td>727.0</td>
      <td>1.0</td>
      <td>50723.0</td>
      <td>34414.0</td>
      <td>0.11</td>
      <td>0.11</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>410.0</td>
      <td>34055.0</td>
      <td>0.041</td>
    </tr>
    <tr>
      <th>20820</th>
      <td>Germany</td>
      <td>DEU</td>
      <td>2020-12-30</td>
      <td>157311.0</td>
      <td>156551.0</td>
      <td>759.0</td>
      <td>1.0</td>
      <td>64129.0</td>
      <td>44319.0</td>
      <td>0.19</td>
      <td>0.19</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>528.0</td>
      <td>44069.0</td>
      <td>0.053</td>
    </tr>
    <tr>
      <th>20821</th>
      <td>Germany</td>
      <td>DEU</td>
      <td>2020-12-31</td>
      <td>207320.0</td>
      <td>206473.0</td>
      <td>846.0</td>
      <td>1.0</td>
      <td>50009.0</td>
      <td>45741.0</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>545.0</td>
      <td>45532.0</td>
      <td>0.054</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig=make_subplots()
fig.add_trace(go.Scatter(x=df_de['date'],y=df_de['people_vaccinated_per_hundred'],name="percentage_people_vaccinated"))
fig.add_trace(go.Scatter(x=df_de['date'],y=df_de['people_fully_vaccinated_per_hundred'],name="percentage_people_fully_vaccinated"))

fig.update_layout(autosize=False,width=900,height=600,title_text="Vaccination in Germany")
fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="Number",secondary_y=False)
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="e04ebb27-5016-415a-a325-8d9cd0134df6" class="plotly-graph-div" style="height:600px; width:900px;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("e04ebb27-5016-415a-a325-8d9cd0134df6")) {
                    Plotly.newPlot(
                        'e04ebb27-5016-415a-a325-8d9cd0134df6',
                        [{"name": "percentage_people_vaccinated", "type": "scatter", "x": ["2020-12-27", "2020-12-28", "2020-12-29", "2020-12-30", "2020-12-31", "2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08", "2021-01-09", "2021-01-10", "2021-01-11", "2021-01-12", "2021-01-13", "2021-01-14", "2021-01-15", "2021-01-16", "2021-01-17", "2021-01-18", "2021-01-19", "2021-01-20", "2021-01-21", "2021-01-22", "2021-01-23", "2021-01-24", "2021-01-25", "2021-01-26", "2021-01-27", "2021-01-28", "2021-01-29", "2021-01-30", "2021-01-31", "2021-02-01", "2021-02-02", "2021-02-03", "2021-02-04", "2021-02-05", "2021-02-06", "2021-02-07", "2021-02-08", "2021-02-09", "2021-02-10", "2021-02-11", "2021-02-12", "2021-02-13", "2021-02-14", "2021-02-15", "2021-02-16", "2021-02-17", "2021-02-18", "2021-02-19", "2021-02-20", "2021-02-21", "2021-02-22", "2021-02-23", "2021-02-24", "2021-02-25", "2021-02-26", "2021-02-27", "2021-02-28", "2021-03-01", "2021-03-02", "2021-03-03", "2021-03-04", "2021-03-05", "2021-03-06", "2021-03-07", "2021-03-08", "2021-03-09", "2021-03-10", "2021-03-11", "2021-03-12", "2021-03-13", "2021-03-14", "2021-03-15", "2021-03-16", "2021-03-17", "2021-03-18", "2021-03-19", "2021-03-20", "2021-03-21", "2021-03-22", "2021-03-23", "2021-03-24", "2021-03-25", "2021-03-26", "2021-03-27", "2021-03-28", "2021-03-29", "2021-03-30", "2021-03-31", "2021-04-01", "2021-04-02", "2021-04-03", "2021-04-04", "2021-04-05", "2021-04-06", "2021-04-07", "2021-04-08", "2021-04-09", "2021-04-10", "2021-04-11", "2021-04-12", "2021-04-13", "2021-04-14", "2021-04-15", "2021-04-16", "2021-04-17", "2021-04-18", "2021-04-19", "2021-04-20", "2021-04-21", "2021-04-22", "2021-04-23", "2021-04-24", "2021-04-25", "2021-04-26", "2021-04-27", "2021-04-28", "2021-04-29", "2021-04-30", "2021-05-01", "2021-05-02", "2021-05-03", "2021-05-04", "2021-05-05", "2021-05-06", "2021-05-07", "2021-05-08", "2021-05-09", "2021-05-10", "2021-05-11", "2021-05-12", "2021-05-13", "2021-05-14", "2021-05-15", "2021-05-16", "2021-05-17", "2021-05-18", "2021-05-19", "2021-05-20", "2021-05-21", "2021-05-22", "2021-05-23", "2021-05-24", "2021-05-25", "2021-05-26", "2021-05-27", "2021-05-28", "2021-05-29", "2021-05-30", "2021-05-31", "2021-06-01", "2021-06-02", "2021-06-03", "2021-06-04", "2021-06-05", "2021-06-06", "2021-06-07", "2021-06-08", "2021-06-09", "2021-06-10", "2021-06-11", "2021-06-12", "2021-06-13", "2021-06-14", "2021-06-15", "2021-06-16", "2021-06-17", "2021-06-18", "2021-06-19", "2021-06-20", "2021-06-21", "2021-06-22", "2021-06-23", "2021-06-24", "2021-06-25", "2021-06-26", "2021-06-27", "2021-06-28", "2021-06-29", "2021-06-30", "2021-07-01", "2021-07-02", "2021-07-03", "2021-07-04", "2021-07-05", "2021-07-06", "2021-07-07", "2021-07-08", "2021-07-09", "2021-07-10", "2021-07-11", "2021-07-12", "2021-07-13", "2021-07-14", "2021-07-15", "2021-07-16", "2021-07-17", "2021-07-18", "2021-07-19", "2021-07-20", "2021-07-21", "2021-07-22", "2021-07-23", "2021-07-24", "2021-07-25", "2021-07-26", "2021-07-27", "2021-07-28", "2021-07-29", "2021-07-30", "2021-07-31", "2021-08-01", "2021-08-02", "2021-08-03", "2021-08-04", "2021-08-05", "2021-08-06", "2021-08-07", "2021-08-08", "2021-08-09", "2021-08-10", "2021-08-11", "2021-08-12", "2021-08-13", "2021-08-14", "2021-08-15", "2021-08-16", "2021-08-17", "2021-08-18", "2021-08-19", "2021-08-20", "2021-08-21", "2021-08-22", "2021-08-23", "2021-08-24", "2021-08-25", "2021-08-26", "2021-08-27", "2021-08-28", "2021-08-29", "2021-08-30", "2021-08-31", "2021-09-01", "2021-09-02", "2021-09-03", "2021-09-04", "2021-09-05", "2021-09-06", "2021-09-07", "2021-09-08", "2021-09-09", "2021-09-10", "2021-09-11", "2021-09-12", "2021-09-13", "2021-09-14", "2021-09-15", "2021-09-16", "2021-09-17", "2021-09-18", "2021-09-19", "2021-09-20", "2021-09-21", "2021-09-22", "2021-09-23", "2021-09-24", "2021-09-25", "2021-09-26", "2021-09-27", "2021-09-28", "2021-09-29", "2021-09-30", "2021-10-01", "2021-10-02", "2021-10-03", "2021-10-04", "2021-10-05", "2021-10-06", "2021-10-07", "2021-10-08", "2021-10-09", "2021-10-10", "2021-10-11", "2021-10-12", "2021-10-13", "2021-10-14", "2021-10-15", "2021-10-16", "2021-10-17", "2021-10-18", "2021-10-19", "2021-10-20", "2021-10-21", "2021-10-22", "2021-10-23", "2021-10-24", "2021-10-25", "2021-10-26", "2021-10-27", "2021-10-28", "2021-10-29", "2021-10-30", "2021-10-31", "2021-11-01", "2021-11-02", "2021-11-03", "2021-11-04", "2021-11-05", "2021-11-06", "2021-11-07", "2021-11-08", "2021-11-09", "2021-11-10", "2021-11-11", "2021-11-12", "2021-11-13", "2021-11-14", "2021-11-15", "2021-11-16"], "y": [0.03, 0.05, 0.11, 0.19, 0.25, 0.28, 0.33, 0.36, 0.42, 0.48, 0.55, 0.62, 0.69, 0.77, 0.81, 0.89, 0.98, 1.12, 1.22, 1.33, 1.39, 1.43, 1.51, 1.61, 1.7, 1.78, 1.86, 1.91, 1.94, 2.0, 2.06, 2.13, 2.18, 2.24, 2.28, 2.31, 2.38, 2.46, 2.54, 2.61, 2.69, 2.75, 2.78, 2.85, 2.92, 3.01, 3.1, 3.2, 3.27, 3.32, 3.4, 3.49, 3.61, 3.72, 3.84, 3.93, 4.0, 4.12, 4.25, 4.39, 4.54, 4.72, 4.85, 4.94, 5.12, 5.32, 5.53, 5.75, 6.0, 6.18, 6.32, 6.54, 6.78, 7.04, 7.31, 7.61, 7.84, 8.0, 8.23, 8.36, 8.51, 8.65, 8.84, 9.04, 9.19, 9.42, 9.68, 9.98, 10.28, 10.62, 10.88, 11.06, 11.31, 11.6, 11.88, 12.16, 12.35, 12.55, 12.72, 12.93, 13.27, 13.95, 14.68, 15.33, 15.69, 15.95, 16.36, 16.96, 17.78, 18.51, 19.14, 19.51, 19.78, 20.19, 20.78, 21.56, 22.24, 22.82, 23.19, 23.47, 23.9, 24.73, 25.92, 26.89, 27.68, 28.0, 28.24, 28.64, 29.45, 30.55, 31.47, 32.26, 32.58, 32.8, 33.28, 34.24, 35.5, 35.82, 36.46, 36.74, 36.94, 37.37, 38.0, 38.68, 39.24, 39.74, 40.01, 40.17, 40.35, 40.71, 41.36, 41.94, 42.46, 42.68, 42.81, 43.07, 43.65, 44.39, 44.73, 45.17, 45.32, 45.42, 45.71, 46.2, 46.86, 47.45, 48.01, 48.2, 48.29, 48.61, 49.1, 49.75, 50.32, 50.81, 50.98, 51.08, 51.37, 51.87, 52.5, 53.13, 53.64, 53.82, 53.93, 54.25, 54.75, 55.31, 55.89, 56.37, 56.57, 56.69, 56.94, 57.29, 57.7, 58.04, 58.33, 58.45, 58.54, 58.74, 58.98, 59.25, 59.48, 59.69, 59.8, 59.88, 60.01, 60.18, 60.38, 60.55, 60.7, 60.78, 60.83, 60.93, 61.06, 61.23, 61.37, 61.5, 61.57, 61.62, 61.7, 61.81, 61.95, 62.07, 62.18, 62.25, 62.28, 62.35, 62.46, 62.6, 62.74, 62.88, 62.96, 63.01, 63.11, 63.24, 63.4, 63.55, 63.71, 63.79, 63.84, 63.95, 64.1, 64.27, 64.43, 64.56, 64.65, 64.69, 64.79, 64.92, 65.07, 65.21, 65.34, 65.39, 65.43, 65.51, 65.62, 65.74, 65.86, 65.97, 66.02, 66.05, 66.13, 66.24, 66.36, 66.49, 66.6, 66.65, 66.68, 66.75, 66.84, 66.95, 67.06, 67.15, 67.19, 67.22, 67.27, 67.36, 67.46, 67.55, 67.6, 67.62, 67.62, 67.66, 67.73, 67.8, 67.87, 67.94, 67.96, 67.97, 68.0, 68.06, 68.13, 68.19, 68.25, 68.27, 68.28, 68.31, 68.36, 68.42, 68.48, 68.53, 68.55, 68.55, 68.58, 68.64, 68.69, 68.75, 68.81, 68.83, 68.83, 68.85, 68.89, 68.95, 69.01, 69.07, 69.09, 69.1, 69.14, 69.2, 69.27, 69.35, 69.41, 69.44, 69.45, 69.5, 69.57]}, {"name": "percentage_people_fully_vaccinated", "type": "scatter", "x": ["2020-12-27", "2020-12-28", "2020-12-29", "2020-12-30", "2020-12-31", "2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08", "2021-01-09", "2021-01-10", "2021-01-11", "2021-01-12", "2021-01-13", "2021-01-14", "2021-01-15", "2021-01-16", "2021-01-17", "2021-01-18", "2021-01-19", "2021-01-20", "2021-01-21", "2021-01-22", "2021-01-23", "2021-01-24", "2021-01-25", "2021-01-26", "2021-01-27", "2021-01-28", "2021-01-29", "2021-01-30", "2021-01-31", "2021-02-01", "2021-02-02", "2021-02-03", "2021-02-04", "2021-02-05", "2021-02-06", "2021-02-07", "2021-02-08", "2021-02-09", "2021-02-10", "2021-02-11", "2021-02-12", "2021-02-13", "2021-02-14", "2021-02-15", "2021-02-16", "2021-02-17", "2021-02-18", "2021-02-19", "2021-02-20", "2021-02-21", "2021-02-22", "2021-02-23", "2021-02-24", "2021-02-25", "2021-02-26", "2021-02-27", "2021-02-28", "2021-03-01", "2021-03-02", "2021-03-03", "2021-03-04", "2021-03-05", "2021-03-06", "2021-03-07", "2021-03-08", "2021-03-09", "2021-03-10", "2021-03-11", "2021-03-12", "2021-03-13", "2021-03-14", "2021-03-15", "2021-03-16", "2021-03-17", "2021-03-18", "2021-03-19", "2021-03-20", "2021-03-21", "2021-03-22", "2021-03-23", "2021-03-24", "2021-03-25", "2021-03-26", "2021-03-27", "2021-03-28", "2021-03-29", "2021-03-30", "2021-03-31", "2021-04-01", "2021-04-02", "2021-04-03", "2021-04-04", "2021-04-05", "2021-04-06", "2021-04-07", "2021-04-08", "2021-04-09", "2021-04-10", "2021-04-11", "2021-04-12", "2021-04-13", "2021-04-14", "2021-04-15", "2021-04-16", "2021-04-17", "2021-04-18", "2021-04-19", "2021-04-20", "2021-04-21", "2021-04-22", "2021-04-23", "2021-04-24", "2021-04-25", "2021-04-26", "2021-04-27", "2021-04-28", "2021-04-29", "2021-04-30", "2021-05-01", "2021-05-02", "2021-05-03", "2021-05-04", "2021-05-05", "2021-05-06", "2021-05-07", "2021-05-08", "2021-05-09", "2021-05-10", "2021-05-11", "2021-05-12", "2021-05-13", "2021-05-14", "2021-05-15", "2021-05-16", "2021-05-17", "2021-05-18", "2021-05-19", "2021-05-20", "2021-05-21", "2021-05-22", "2021-05-23", "2021-05-24", "2021-05-25", "2021-05-26", "2021-05-27", "2021-05-28", "2021-05-29", "2021-05-30", "2021-05-31", "2021-06-01", "2021-06-02", "2021-06-03", "2021-06-04", "2021-06-05", "2021-06-06", "2021-06-07", "2021-06-08", "2021-06-09", "2021-06-10", "2021-06-11", "2021-06-12", "2021-06-13", "2021-06-14", "2021-06-15", "2021-06-16", "2021-06-17", "2021-06-18", "2021-06-19", "2021-06-20", "2021-06-21", "2021-06-22", "2021-06-23", "2021-06-24", "2021-06-25", "2021-06-26", "2021-06-27", "2021-06-28", "2021-06-29", "2021-06-30", "2021-07-01", "2021-07-02", "2021-07-03", "2021-07-04", "2021-07-05", "2021-07-06", "2021-07-07", "2021-07-08", "2021-07-09", "2021-07-10", "2021-07-11", "2021-07-12", "2021-07-13", "2021-07-14", "2021-07-15", "2021-07-16", "2021-07-17", "2021-07-18", "2021-07-19", "2021-07-20", "2021-07-21", "2021-07-22", "2021-07-23", "2021-07-24", "2021-07-25", "2021-07-26", "2021-07-27", "2021-07-28", "2021-07-29", "2021-07-30", "2021-07-31", "2021-08-01", "2021-08-02", "2021-08-03", "2021-08-04", "2021-08-05", "2021-08-06", "2021-08-07", "2021-08-08", "2021-08-09", "2021-08-10", "2021-08-11", "2021-08-12", "2021-08-13", "2021-08-14", "2021-08-15", "2021-08-16", "2021-08-17", "2021-08-18", "2021-08-19", "2021-08-20", "2021-08-21", "2021-08-22", "2021-08-23", "2021-08-24", "2021-08-25", "2021-08-26", "2021-08-27", "2021-08-28", "2021-08-29", "2021-08-30", "2021-08-31", "2021-09-01", "2021-09-02", "2021-09-03", "2021-09-04", "2021-09-05", "2021-09-06", "2021-09-07", "2021-09-08", "2021-09-09", "2021-09-10", "2021-09-11", "2021-09-12", "2021-09-13", "2021-09-14", "2021-09-15", "2021-09-16", "2021-09-17", "2021-09-18", "2021-09-19", "2021-09-20", "2021-09-21", "2021-09-22", "2021-09-23", "2021-09-24", "2021-09-25", "2021-09-26", "2021-09-27", "2021-09-28", "2021-09-29", "2021-09-30", "2021-10-01", "2021-10-02", "2021-10-03", "2021-10-04", "2021-10-05", "2021-10-06", "2021-10-07", "2021-10-08", "2021-10-09", "2021-10-10", "2021-10-11", "2021-10-12", "2021-10-13", "2021-10-14", "2021-10-15", "2021-10-16", "2021-10-17", "2021-10-18", "2021-10-19", "2021-10-20", "2021-10-21", "2021-10-22", "2021-10-23", "2021-10-24", "2021-10-25", "2021-10-26", "2021-10-27", "2021-10-28", "2021-10-29", "2021-10-30", "2021-10-31", "2021-11-01", "2021-11-02", "2021-11-03", "2021-11-04", "2021-11-05", "2021-11-06", "2021-11-07", "2021-11-08", "2021-11-09", "2021-11-10", "2021-11-11", "2021-11-12", "2021-11-13", "2021-11-14", "2021-11-15", "2021-11-16"], "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.03, 0.04, 0.09, 0.15, 0.2, 0.24, 0.3, 0.32, 0.37, 0.42, 0.49, 0.55, 0.62, 0.68, 0.72, 0.79, 0.87, 0.99, 1.08, 1.17, 1.23, 1.26, 1.33, 1.41, 1.51, 1.59, 1.68, 1.73, 1.76, 1.83, 1.9, 1.97, 2.03, 2.1, 2.14, 2.17, 2.24, 2.31, 2.39, 2.46, 2.54, 2.58, 2.61, 2.67, 2.74, 2.82, 2.89, 2.98, 3.03, 3.07, 3.14, 3.2, 3.28, 3.35, 3.44, 3.5, 3.54, 3.61, 3.69, 3.79, 3.88, 3.98, 4.04, 4.09, 4.18, 4.28, 4.39, 4.49, 4.59, 4.67, 4.73, 4.84, 4.95, 5.07, 5.18, 5.26, 5.34, 5.4, 5.48, 5.59, 5.71, 5.81, 5.92, 6.0, 6.06, 6.15, 6.24, 6.34, 6.42, 6.51, 6.58, 6.62, 6.7, 6.77, 6.86, 6.94, 7.02, 7.09, 7.15, 7.24, 7.35, 7.5, 7.66, 7.82, 7.92, 8.0, 8.14, 8.33, 8.61, 8.87, 9.16, 9.34, 9.47, 9.7, 10.04, 10.48, 10.67, 10.95, 11.14, 11.29, 11.53, 11.96, 12.6, 13.18, 13.72, 14.02, 14.22, 14.44, 14.89, 15.74, 16.5, 17.15, 17.46, 17.68, 18.12, 18.83, 19.7, 20.19, 20.81, 21.19, 21.47, 21.97, 22.82, 23.98, 24.92, 25.75, 26.12, 26.38, 26.91, 27.74, 28.84, 29.73, 30.5, 30.87, 31.14, 31.64, 32.48, 33.49, 34.14, 34.78, 35.11, 35.35, 35.81, 36.46, 37.14, 37.77, 38.32, 38.6, 38.81, 39.22, 39.82, 40.63, 41.32, 41.97, 42.26, 42.46, 42.84, 43.53, 44.46, 45.17, 45.78, 46.02, 46.18, 46.53, 47.11, 47.82, 48.42, 48.97, 49.16, 49.27, 49.59, 50.13, 50.81, 51.38, 51.86, 52.03, 52.14, 52.42, 52.88, 53.44, 53.96, 54.39, 54.56, 54.67, 54.96, 55.39, 55.91, 56.36, 56.75, 56.92, 57.02, 57.26, 57.57, 57.93, 58.21, 58.49, 58.6, 58.67, 58.86, 59.09, 59.35, 59.57, 59.75, 59.85, 59.91, 60.06, 60.23, 60.43, 60.6, 60.76, 60.84, 60.89, 60.99, 61.13, 61.3, 61.46, 61.62, 61.7, 61.75, 61.86, 62.01, 62.19, 62.36, 62.53, 62.62, 62.67, 62.78, 62.92, 63.09, 63.26, 63.42, 63.5, 63.54, 63.64, 63.77, 63.93, 64.08, 64.16, 64.18, 64.19, 64.24, 64.35, 64.47, 64.59, 64.7, 64.72, 64.74, 64.79, 64.89, 65.0, 65.11, 65.21, 65.23, 65.24, 65.29, 65.38, 65.48, 65.58, 65.66, 65.69, 65.69, 65.74, 65.84, 65.94, 66.04, 66.12, 66.15, 66.15, 66.18, 66.25, 66.35, 66.44, 66.51, 66.53, 66.55, 66.59, 66.68, 66.77, 66.86, 66.93, 66.95, 66.96, 67.01, 67.08]}],
                        {"autosize": false, "height": 600, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Vaccination in Germany"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "Date"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "Number"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('e04ebb27-5016-415a-a325-8d9cd0134df6');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


Nun sind ungefähr 67,7% der deutschen Gesamtbevölkerung vollständig geimpft (17.11.2021)

# Covid Death


```python
! kaggle datasets  download -d  dhruvildave/covid19-deaths-dataset
```


```python

```


```python
! mkdir train
! unzip covid19-deaths-dataset.zip -d train
```

    mkdir: cannot create directory ‘train’: File exists
    Archive:  covid19-deaths-dataset.zip
      inflating: train/all_weekly_excess_deaths.csv  
      inflating: train/us-counties.csv   
    

EDA


```python
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from collections import Counter
import operator
```

## Data Statistics Exploration


```python
df = pd.read_csv("train/all_weekly_excess_deaths.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>region</th>
      <th>region_code</th>
      <th>start_date</th>
      <th>end_date</th>
      <th>days</th>
      <th>year</th>
      <th>week</th>
      <th>population</th>
      <th>total_deaths</th>
      <th>covid_deaths</th>
      <th>expected_deaths</th>
      <th>excess_deaths</th>
      <th>non_covid_deaths</th>
      <th>covid_deaths_per_100k</th>
      <th>excess_deaths_per_100k</th>
      <th>excess_deaths_pct_change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Australia</td>
      <td>Australia</td>
      <td>0</td>
      <td>2019-12-30</td>
      <td>2020-01-05</td>
      <td>7</td>
      <td>2020</td>
      <td>1</td>
      <td>25788217</td>
      <td>2510.0</td>
      <td>0.0</td>
      <td>2569.892790</td>
      <td>-59.892790</td>
      <td>2510.0</td>
      <td>0.0</td>
      <td>-0.232249</td>
      <td>-0.023306</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Australia</td>
      <td>Australia</td>
      <td>0</td>
      <td>2020-01-06</td>
      <td>2020-01-12</td>
      <td>7</td>
      <td>2020</td>
      <td>2</td>
      <td>25788217</td>
      <td>2523.0</td>
      <td>0.0</td>
      <td>2565.059457</td>
      <td>-42.059457</td>
      <td>2523.0</td>
      <td>0.0</td>
      <td>-0.163096</td>
      <td>-0.016397</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Australia</td>
      <td>Australia</td>
      <td>0</td>
      <td>2020-01-13</td>
      <td>2020-01-19</td>
      <td>7</td>
      <td>2020</td>
      <td>3</td>
      <td>25788217</td>
      <td>2516.0</td>
      <td>0.0</td>
      <td>2543.559457</td>
      <td>-27.559457</td>
      <td>2516.0</td>
      <td>0.0</td>
      <td>-0.106868</td>
      <td>-0.010835</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Australia</td>
      <td>Australia</td>
      <td>0</td>
      <td>2020-01-20</td>
      <td>2020-01-26</td>
      <td>7</td>
      <td>2020</td>
      <td>4</td>
      <td>25788217</td>
      <td>2619.0</td>
      <td>0.0</td>
      <td>2544.892790</td>
      <td>74.107210</td>
      <td>2619.0</td>
      <td>0.0</td>
      <td>0.287368</td>
      <td>0.029120</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Australia</td>
      <td>Australia</td>
      <td>0</td>
      <td>2020-01-27</td>
      <td>2020-02-02</td>
      <td>7</td>
      <td>2020</td>
      <td>5</td>
      <td>25788217</td>
      <td>2522.0</td>
      <td>0.0</td>
      <td>2532.392790</td>
      <td>-10.392790</td>
      <td>2522.0</td>
      <td>0.0</td>
      <td>-0.040301</td>
      <td>-0.004104</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info() # no missing values found
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8630 entries, 0 to 8629
    Data columns (total 17 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   country                   8630 non-null   object 
     1   region                    8630 non-null   object 
     2   region_code               8630 non-null   object 
     3   start_date                8630 non-null   object 
     4   end_date                  8630 non-null   object 
     5   days                      8630 non-null   int64  
     6   year                      8630 non-null   int64  
     7   week                      8630 non-null   int64  
     8   population                8630 non-null   int64  
     9   total_deaths              8630 non-null   float64
     10  covid_deaths              8630 non-null   float64
     11  expected_deaths           8630 non-null   float64
     12  excess_deaths             8630 non-null   float64
     13  non_covid_deaths          8630 non-null   float64
     14  covid_deaths_per_100k     8630 non-null   float64
     15  excess_deaths_per_100k    8630 non-null   float64
     16  excess_deaths_pct_change  8630 non-null   float64
    dtypes: float64(8), int64(4), object(5)
    memory usage: 1.1+ MB
    


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>days</th>
      <th>year</th>
      <th>week</th>
      <th>population</th>
      <th>total_deaths</th>
      <th>covid_deaths</th>
      <th>expected_deaths</th>
      <th>excess_deaths</th>
      <th>non_covid_deaths</th>
      <th>covid_deaths_per_100k</th>
      <th>excess_deaths_per_100k</th>
      <th>excess_deaths_pct_change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8630.000000</td>
      <td>8630.000000</td>
      <td>8630.000000</td>
      <td>8.630000e+03</td>
      <td>8630.000000</td>
      <td>8630.000000</td>
      <td>8630.000000</td>
      <td>8630.000000</td>
      <td>8630.000000</td>
      <td>8630.000000</td>
      <td>8630.000000</td>
      <td>8630.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.999421</td>
      <td>2020.398378</td>
      <td>23.668366</td>
      <td>1.814569e+07</td>
      <td>3344.657451</td>
      <td>357.936443</td>
      <td>2861.936587</td>
      <td>482.720864</td>
      <td>2986.721008</td>
      <td>1.865979</td>
      <td>2.406849</td>
      <td>0.147073</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.053823</td>
      <td>0.489592</td>
      <td>14.283964</td>
      <td>3.830643e+07</td>
      <td>7431.723548</td>
      <td>1179.567170</td>
      <td>6252.342704</td>
      <td>1727.999538</td>
      <td>6564.975068</td>
      <td>2.861617</td>
      <td>3.884268</td>
      <td>0.248748</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.000000</td>
      <td>2020.000000</td>
      <td>1.000000</td>
      <td>3.433600e+05</td>
      <td>28.000000</td>
      <td>-1625.000000</td>
      <td>36.958708</td>
      <td>-3900.712360</td>
      <td>-1740.000000</td>
      <td>-8.803323</td>
      <td>-8.774060</td>
      <td>-0.450265</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.000000</td>
      <td>2020.000000</td>
      <td>12.000000</td>
      <td>2.689862e+06</td>
      <td>550.000000</td>
      <td>4.000000</td>
      <td>494.844756</td>
      <td>3.459340</td>
      <td>503.000000</td>
      <td>0.113038</td>
      <td>0.109435</td>
      <td>0.006566</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.000000</td>
      <td>2020.000000</td>
      <td>23.000000</td>
      <td>6.732219e+06</td>
      <td>1248.000000</td>
      <td>44.000000</td>
      <td>1116.339077</td>
      <td>68.225000</td>
      <td>1123.000000</td>
      <td>0.815235</td>
      <td>1.424369</td>
      <td>0.085771</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.000000</td>
      <td>2021.000000</td>
      <td>34.000000</td>
      <td>1.717309e+07</td>
      <td>2679.000000</td>
      <td>210.000000</td>
      <td>2367.852564</td>
      <td>281.271795</td>
      <td>2404.750000</td>
      <td>2.356389</td>
      <td>3.432663</td>
      <td>0.206313</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.000000</td>
      <td>2021.000000</td>
      <td>53.000000</td>
      <td>3.283005e+08</td>
      <td>87342.000000</td>
      <td>23481.000000</td>
      <td>62621.817308</td>
      <td>27935.009878</td>
      <td>70474.000000</td>
      <td>43.366504</td>
      <td>48.776239</td>
      <td>3.759663</td>
    </tr>
  </tbody>
</table>
</div>




```python
# we plot numbe of entries in the dataset
fig, ax = plt.subplots(1,1,figsize=(15,5))
sns.countplot(data=df,x='country',ax=ax)
plt.xticks(rotation=90)
plt.show()
```


    
![png](/_posts/covid/output_34_0.png)
    



```python
#  plot the total number of death cases
fig,ax = plt.subplots(1,1,figsize=(15,5))
sns.barplot(data=df,x='country',y='covid_deaths')
plt.xticks(rotation=60)
plt.show()
```


    
![png](/_posts/covid/output_35_0.png)
    



```python
#  plot the percentage of COVID death cases
fig,ax =plt.subplots(1,1,figsize=(15,5))
df["covid_death_percent"] = df["covid_deaths"]/df["total_deaths"] * 100
sns.barplot(data=df,x='country',y='covid_death_percent')
plt.xticks(rotation=90)
plt.show()
```


    
![png](/_posts/covid/output_36_0.png)
    


Mexico has the highest number of Covid death and Peru has the highest perentage of Covid death


```python
# Covid deaths over the time period
fig = px.choropleth(data_frame=df, locations='country',
                    locationmode='country names', color='covid_deaths',
                    animation_frame='end_date')
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="36958287-aa3e-41fb-81f8-e2a8da9f0203" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("36958287-aa3e-41fb-81f8-e2a8da9f0203")) {
                    Plotly.newPlot(
                        '36958287-aa3e-41fb-81f8-e2a8da9f0203',
                        [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-01-05<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "type": "choropleth", "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}],
                        {"coloraxis": {"colorbar": {"title": {"text": "covid_deaths"}}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "geo": {"center": {}, "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]}}, "legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "sliders": [{"active": 0, "currentvalue": {"prefix": "end_date="}, "len": 0.9, "pad": {"b": 10, "t": 60}, "steps": [{"args": [["2020-01-05"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-01-05", "method": "animate"}, {"args": [["2020-01-12"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-01-12", "method": "animate"}, {"args": [["2020-01-19"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-01-19", "method": "animate"}, {"args": [["2020-01-26"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-01-26", "method": "animate"}, {"args": [["2020-02-02"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-02-02", "method": "animate"}, {"args": [["2020-02-09"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-02-09", "method": "animate"}, {"args": [["2020-02-16"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-02-16", "method": "animate"}, {"args": [["2020-02-23"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-02-23", "method": "animate"}, {"args": [["2020-03-01"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-03-01", "method": "animate"}, {"args": [["2020-03-08"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-03-08", "method": "animate"}, {"args": [["2020-03-15"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-03-15", "method": "animate"}, {"args": [["2020-03-22"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-03-22", "method": "animate"}, {"args": [["2020-03-29"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-03-29", "method": "animate"}, {"args": [["2020-04-05"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-04-05", "method": "animate"}, {"args": [["2020-04-12"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-04-12", "method": "animate"}, {"args": [["2020-04-19"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-04-19", "method": "animate"}, {"args": [["2020-04-26"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-04-26", "method": "animate"}, {"args": [["2020-05-03"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-05-03", "method": "animate"}, {"args": [["2020-05-10"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-05-10", "method": "animate"}, {"args": [["2020-05-17"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-05-17", "method": "animate"}, {"args": [["2020-05-24"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-05-24", "method": "animate"}, {"args": [["2020-05-31"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-05-31", "method": "animate"}, {"args": [["2020-06-07"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-06-07", "method": "animate"}, {"args": [["2020-06-14"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-06-14", "method": "animate"}, {"args": [["2020-06-21"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-06-21", "method": "animate"}, {"args": [["2020-06-28"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-06-28", "method": "animate"}, {"args": [["2020-07-05"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-07-05", "method": "animate"}, {"args": [["2020-07-12"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-07-12", "method": "animate"}, {"args": [["2020-07-19"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-07-19", "method": "animate"}, {"args": [["2020-07-26"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-07-26", "method": "animate"}, {"args": [["2020-08-02"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-08-02", "method": "animate"}, {"args": [["2020-08-09"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-08-09", "method": "animate"}, {"args": [["2020-08-16"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-08-16", "method": "animate"}, {"args": [["2020-08-23"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-08-23", "method": "animate"}, {"args": [["2020-08-30"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-08-30", "method": "animate"}, {"args": [["2020-09-06"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-09-06", "method": "animate"}, {"args": [["2020-09-13"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-09-13", "method": "animate"}, {"args": [["2020-09-20"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-09-20", "method": "animate"}, {"args": [["2020-09-27"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-09-27", "method": "animate"}, {"args": [["2020-10-04"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-10-04", "method": "animate"}, {"args": [["2020-10-11"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-10-11", "method": "animate"}, {"args": [["2020-10-18"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-10-18", "method": "animate"}, {"args": [["2020-10-25"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-10-25", "method": "animate"}, {"args": [["2020-11-01"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-11-01", "method": "animate"}, {"args": [["2020-11-08"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-11-08", "method": "animate"}, {"args": [["2020-11-15"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-11-15", "method": "animate"}, {"args": [["2020-11-22"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-11-22", "method": "animate"}, {"args": [["2020-11-29"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-11-29", "method": "animate"}, {"args": [["2020-12-06"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-12-06", "method": "animate"}, {"args": [["2020-12-13"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-12-13", "method": "animate"}, {"args": [["2020-12-20"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-12-20", "method": "animate"}, {"args": [["2020-12-27"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-12-27", "method": "animate"}, {"args": [["2021-01-03"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-01-03", "method": "animate"}, {"args": [["2021-01-10"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-01-10", "method": "animate"}, {"args": [["2021-01-17"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-01-17", "method": "animate"}, {"args": [["2021-01-24"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-01-24", "method": "animate"}, {"args": [["2021-01-31"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-01-31", "method": "animate"}, {"args": [["2021-02-07"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-02-07", "method": "animate"}, {"args": [["2021-02-14"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-02-14", "method": "animate"}, {"args": [["2021-02-21"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-02-21", "method": "animate"}, {"args": [["2021-02-28"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-02-28", "method": "animate"}, {"args": [["2021-03-07"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-03-07", "method": "animate"}, {"args": [["2021-03-14"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-03-14", "method": "animate"}, {"args": [["2021-03-21"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-03-21", "method": "animate"}, {"args": [["2021-03-28"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-03-28", "method": "animate"}, {"args": [["2021-04-04"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-04-04", "method": "animate"}, {"args": [["2021-04-11"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-04-11", "method": "animate"}, {"args": [["2021-04-18"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-04-18", "method": "animate"}, {"args": [["2021-04-25"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-04-25", "method": "animate"}, {"args": [["2021-05-02"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-05-02", "method": "animate"}, {"args": [["2021-05-09"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-05-09", "method": "animate"}, {"args": [["2021-05-16"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-05-16", "method": "animate"}, {"args": [["2021-05-23"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-05-23", "method": "animate"}, {"args": [["2021-05-30"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-05-30", "method": "animate"}, {"args": [["2021-06-06"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-06-06", "method": "animate"}, {"args": [["2021-06-13"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-06-13", "method": "animate"}, {"args": [["2021-06-20"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-06-20", "method": "animate"}, {"args": [["2021-06-27"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-06-27", "method": "animate"}, {"args": [["2021-07-04"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-07-04", "method": "animate"}, {"args": [["2021-07-11"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-07-11", "method": "animate"}, {"args": [["2021-07-18"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-07-18", "method": "animate"}, {"args": [["2021-07-25"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-07-25", "method": "animate"}, {"args": [["2021-08-01"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-08-01", "method": "animate"}, {"args": [["2021-08-08"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-08-08", "method": "animate"}, {"args": [["2021-08-15"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-08-15", "method": "animate"}, {"args": [["2021-08-22"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-08-22", "method": "animate"}, {"args": [["2021-08-29"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-08-29", "method": "animate"}, {"args": [["2021-09-05"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-09-05", "method": "animate"}, {"args": [["2021-09-12"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-09-12", "method": "animate"}, {"args": [["2021-09-19"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-09-19", "method": "animate"}, {"args": [["2021-09-26"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-09-26", "method": "animate"}, {"args": [["2021-10-03"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-10-03", "method": "animate"}, {"args": [["2021-10-10"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-10-10", "method": "animate"}, {"args": [["2021-10-17"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-10-17", "method": "animate"}, {"args": [["2021-10-24"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-10-24", "method": "animate"}, {"args": [["2021-10-31"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-10-31", "method": "animate"}, {"args": [["2021-11-07"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-11-07", "method": "animate"}, {"args": [["2020-01-08"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-01-08", "method": "animate"}, {"args": [["2020-01-15"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-01-15", "method": "animate"}, {"args": [["2020-01-22"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-01-22", "method": "animate"}, {"args": [["2020-01-29"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-01-29", "method": "animate"}, {"args": [["2020-02-05"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-02-05", "method": "animate"}, {"args": [["2020-02-12"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-02-12", "method": "animate"}, {"args": [["2020-02-19"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-02-19", "method": "animate"}, {"args": [["2020-02-26"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-02-26", "method": "animate"}, {"args": [["2020-03-04"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-03-04", "method": "animate"}, {"args": [["2020-03-11"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-03-11", "method": "animate"}, {"args": [["2020-03-18"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-03-18", "method": "animate"}, {"args": [["2020-03-25"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-03-25", "method": "animate"}, {"args": [["2020-04-01"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-04-01", "method": "animate"}, {"args": [["2020-04-08"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-04-08", "method": "animate"}, {"args": [["2020-04-15"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-04-15", "method": "animate"}, {"args": [["2020-04-22"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-04-22", "method": "animate"}, {"args": [["2020-04-29"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-04-29", "method": "animate"}, {"args": [["2020-05-06"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-05-06", "method": "animate"}, {"args": [["2020-05-13"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-05-13", "method": "animate"}, {"args": [["2020-05-20"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-05-20", "method": "animate"}, {"args": [["2020-05-27"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-05-27", "method": "animate"}, {"args": [["2020-06-03"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-06-03", "method": "animate"}, {"args": [["2020-06-10"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-06-10", "method": "animate"}, {"args": [["2020-06-17"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-06-17", "method": "animate"}, {"args": [["2020-06-24"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-06-24", "method": "animate"}, {"args": [["2020-07-01"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-07-01", "method": "animate"}, {"args": [["2020-07-08"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-07-08", "method": "animate"}, {"args": [["2020-07-15"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-07-15", "method": "animate"}, {"args": [["2020-07-22"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-07-22", "method": "animate"}, {"args": [["2020-07-29"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-07-29", "method": "animate"}, {"args": [["2020-08-05"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-08-05", "method": "animate"}, {"args": [["2020-08-12"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-08-12", "method": "animate"}, {"args": [["2020-08-19"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-08-19", "method": "animate"}, {"args": [["2020-08-26"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-08-26", "method": "animate"}, {"args": [["2020-09-02"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-09-02", "method": "animate"}, {"args": [["2020-09-09"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-09-09", "method": "animate"}, {"args": [["2020-09-16"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-09-16", "method": "animate"}, {"args": [["2020-09-23"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-09-23", "method": "animate"}, {"args": [["2020-09-30"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-09-30", "method": "animate"}, {"args": [["2020-10-07"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-10-07", "method": "animate"}, {"args": [["2020-10-14"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-10-14", "method": "animate"}, {"args": [["2020-10-21"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-10-21", "method": "animate"}, {"args": [["2020-10-28"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-10-28", "method": "animate"}, {"args": [["2020-11-04"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-11-04", "method": "animate"}, {"args": [["2020-11-11"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-11-11", "method": "animate"}, {"args": [["2020-11-18"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-11-18", "method": "animate"}, {"args": [["2020-11-25"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-11-25", "method": "animate"}, {"args": [["2020-12-02"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-12-02", "method": "animate"}, {"args": [["2020-12-09"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-12-09", "method": "animate"}, {"args": [["2020-12-16"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-12-16", "method": "animate"}, {"args": [["2020-12-23"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-12-23", "method": "animate"}, {"args": [["2020-12-30"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-12-30", "method": "animate"}, {"args": [["2021-01-01"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-01-01", "method": "animate"}, {"args": [["2020-01-04"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-01-04", "method": "animate"}, {"args": [["2020-01-11"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-01-11", "method": "animate"}, {"args": [["2020-01-18"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-01-18", "method": "animate"}, {"args": [["2020-01-25"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-01-25", "method": "animate"}, {"args": [["2020-02-01"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-02-01", "method": "animate"}, {"args": [["2020-02-08"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-02-08", "method": "animate"}, {"args": [["2020-02-15"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-02-15", "method": "animate"}, {"args": [["2020-02-22"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-02-22", "method": "animate"}, {"args": [["2020-02-29"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-02-29", "method": "animate"}, {"args": [["2020-03-07"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-03-07", "method": "animate"}, {"args": [["2020-03-14"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-03-14", "method": "animate"}, {"args": [["2020-03-21"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-03-21", "method": "animate"}, {"args": [["2020-03-28"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-03-28", "method": "animate"}, {"args": [["2020-04-04"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-04-04", "method": "animate"}, {"args": [["2020-04-11"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-04-11", "method": "animate"}, {"args": [["2020-04-18"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-04-18", "method": "animate"}, {"args": [["2020-04-25"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-04-25", "method": "animate"}, {"args": [["2020-05-02"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-05-02", "method": "animate"}, {"args": [["2020-05-09"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-05-09", "method": "animate"}, {"args": [["2020-05-16"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-05-16", "method": "animate"}, {"args": [["2020-05-23"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-05-23", "method": "animate"}, {"args": [["2020-05-30"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-05-30", "method": "animate"}, {"args": [["2020-06-06"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-06-06", "method": "animate"}, {"args": [["2020-06-13"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-06-13", "method": "animate"}, {"args": [["2020-06-20"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-06-20", "method": "animate"}, {"args": [["2020-06-27"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-06-27", "method": "animate"}, {"args": [["2020-07-04"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-07-04", "method": "animate"}, {"args": [["2020-07-11"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-07-11", "method": "animate"}, {"args": [["2020-07-18"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-07-18", "method": "animate"}, {"args": [["2020-07-25"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-07-25", "method": "animate"}, {"args": [["2020-08-01"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-08-01", "method": "animate"}, {"args": [["2020-08-08"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-08-08", "method": "animate"}, {"args": [["2020-08-15"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-08-15", "method": "animate"}, {"args": [["2020-08-22"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-08-22", "method": "animate"}, {"args": [["2020-08-29"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-08-29", "method": "animate"}, {"args": [["2020-09-05"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-09-05", "method": "animate"}, {"args": [["2020-09-12"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-09-12", "method": "animate"}, {"args": [["2020-09-19"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-09-19", "method": "animate"}, {"args": [["2020-09-26"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-09-26", "method": "animate"}, {"args": [["2020-10-03"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-10-03", "method": "animate"}, {"args": [["2020-10-10"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-10-10", "method": "animate"}, {"args": [["2020-10-17"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-10-17", "method": "animate"}, {"args": [["2020-10-24"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-10-24", "method": "animate"}, {"args": [["2020-10-31"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-10-31", "method": "animate"}, {"args": [["2020-11-07"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-11-07", "method": "animate"}, {"args": [["2020-11-14"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-11-14", "method": "animate"}, {"args": [["2020-11-21"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-11-21", "method": "animate"}, {"args": [["2020-11-28"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-11-28", "method": "animate"}, {"args": [["2020-12-05"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-12-05", "method": "animate"}, {"args": [["2020-12-12"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-12-12", "method": "animate"}, {"args": [["2020-12-19"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-12-19", "method": "animate"}, {"args": [["2020-12-26"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2020-12-26", "method": "animate"}, {"args": [["2021-01-02"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-01-02", "method": "animate"}, {"args": [["2021-01-09"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-01-09", "method": "animate"}, {"args": [["2021-01-16"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-01-16", "method": "animate"}, {"args": [["2021-01-23"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-01-23", "method": "animate"}, {"args": [["2021-01-30"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-01-30", "method": "animate"}, {"args": [["2021-02-06"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-02-06", "method": "animate"}, {"args": [["2021-02-13"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-02-13", "method": "animate"}, {"args": [["2021-02-20"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-02-20", "method": "animate"}, {"args": [["2021-02-27"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-02-27", "method": "animate"}, {"args": [["2021-03-06"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-03-06", "method": "animate"}, {"args": [["2021-03-13"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-03-13", "method": "animate"}, {"args": [["2021-03-20"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-03-20", "method": "animate"}, {"args": [["2021-03-27"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-03-27", "method": "animate"}, {"args": [["2021-04-03"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-04-03", "method": "animate"}, {"args": [["2021-04-10"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-04-10", "method": "animate"}, {"args": [["2021-04-17"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-04-17", "method": "animate"}, {"args": [["2021-04-24"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-04-24", "method": "animate"}, {"args": [["2021-05-01"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-05-01", "method": "animate"}, {"args": [["2021-05-08"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-05-08", "method": "animate"}, {"args": [["2021-05-15"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-05-15", "method": "animate"}, {"args": [["2021-05-22"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-05-22", "method": "animate"}, {"args": [["2021-05-29"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-05-29", "method": "animate"}, {"args": [["2021-06-05"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-06-05", "method": "animate"}, {"args": [["2021-06-12"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-06-12", "method": "animate"}, {"args": [["2021-06-19"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-06-19", "method": "animate"}, {"args": [["2021-06-26"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-06-26", "method": "animate"}, {"args": [["2021-07-03"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-07-03", "method": "animate"}, {"args": [["2021-07-10"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-07-10", "method": "animate"}, {"args": [["2021-07-17"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-07-17", "method": "animate"}, {"args": [["2021-07-24"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-07-24", "method": "animate"}, {"args": [["2021-07-31"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-07-31", "method": "animate"}, {"args": [["2021-08-07"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-08-07", "method": "animate"}, {"args": [["2021-08-14"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-08-14", "method": "animate"}, {"args": [["2021-08-21"], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "2021-08-21", "method": "animate"}], "x": 0.1, "xanchor": "left", "y": 0, "yanchor": "top"}], "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "updatemenus": [{"buttons": [{"args": [null, {"frame": {"duration": 500, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 500, "easing": "linear"}}], "label": "&#9654;", "method": "animate"}, {"args": [[null], {"frame": {"duration": 0, "redraw": true}, "fromcurrent": true, "mode": "immediate", "transition": {"duration": 0, "easing": "linear"}}], "label": "&#9724;", "method": "animate"}], "direction": "left", "pad": {"r": 10, "t": 70}, "showactive": false, "type": "buttons", "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top"}]},
                        {"responsive": true}
                    ).then(function(){
                            Plotly.addFrames('36958287-aa3e-41fb-81f8-e2a8da9f0203', [{"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-01-05<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "type": "choropleth"}], "name": "2020-01-05"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-01-12<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "type": "choropleth"}], "name": "2020-01-12"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-01-19<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "type": "choropleth"}], "name": "2020-01-19"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-01-26<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "type": "choropleth"}], "name": "2020-01-26"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-02-02<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "type": "choropleth"}], "name": "2020-02-02"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-02-09<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "type": "choropleth"}], "name": "2020-02-09"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-02-16<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "type": "choropleth"}], "name": "2020-02-16"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-02-23<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0], "type": "choropleth"}], "name": "2020-02-23"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-03-01<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 46.0, 0.0, 31.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 0.0], "type": "choropleth"}], "name": "2020-03-01"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-03-08<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 140.0, 0.0, 332.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 33.0, 17.0, 0.0, 2.0, 0.0], "type": "choropleth"}], "name": "2020-03-08"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-03-15<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 1.0, 4.0, 41.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 72.0, 11.0, 4.0, 0.0, 1.0, 0.0, 530.0, 0.0, 1443.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 17.0, 0.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 25.0, 272.0, 7.0, 20.0, 0.0], "type": "choropleth"}], "name": "2020-03-15"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-03-22<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [4.0, 15.0, 71.0, 246.0, 1.0, 21.0, 1.0, 2.0, 1.0, 0.0, 1.0, 11.0, 12.0, 0.0, 1.0, 585.0, 83.0, 11.0, 1.0, 5.0, 1.0, 961.0, 1.0, 3667.0, 0.0, 0.0, 7.0, 0.0, 3.0, 0.0, 160.0, 0.0, 4.0, 29.0, 4.0, 14.0, 3.0, 0.0, 1.0, 0.0, 36.0, 1483.0, 53.0, 91.0, 3.0], "type": "choropleth"}], "name": "2020-03-22"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-03-29<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [9.0, 70.0, 356.0, 1390.0, 5.0, 74.0, 6.0, 8.0, 5.0, 5.0, 15.0, 59.0, 44.0, 3.0, 10.0, 1932.0, 439.0, 23.0, 0.0, 7.0, 1.0, 955.0, 15.0, 5303.0, 0.0, 4.0, 13.0, 0.0, 17.0, 1.0, 592.0, 1.0, 18.0, 61.0, 15.0, 105.0, 40.0, 0.0, 9.0, 2.0, 41.0, 5031.0, 224.0, 263.0, 5.0], "type": "choropleth"}], "name": "2020-03-29"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-04-05<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [19.0, 118.0, 1016.0, 4203.0, 12.0, 339.0, 27.0, 25.0, 9.0, 4.0, 51.0, 107.0, 122.0, 12.0, 17.0, 5473.0, 1051.0, 35.0, 1.0, 21.0, 2.0, 963.0, 40.0, 5108.0, 1.0, 4.0, 15.0, 0.0, 74.0, 1.0, 999.0, 0.0, 46.0, 222.0, 72.0, 176.0, 108.0, 1.0, 17.0, 9.0, 31.0, 5838.0, 117.0, 395.0, 14.0], "type": "choropleth"}], "name": "2020-04-05"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-04-12<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [25.0, 146.0, 2153.0, 6448.0, 9.0, 660.0, 46.0, 74.0, 8.0, 1.0, 71.0, 94.0, 153.0, 10.0, 28.0, 6315.0, 1438.0, 20.0, 3.0, 65.0, 4.0, 871.0, 53.0, 4012.0, 4.0, 6.0, 30.0, 3.0, 202.0, 1.0, 976.0, 3.0, 57.0, 529.0, 138.0, 209.0, 165.0, 1.0, 25.0, 14.0, 31.0, 4568.0, 498.0, 370.0, 9.0], "type": "choropleth"}], "name": "2020-04-12"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-04-19<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [7.0, 102.0, 2083.0, 6184.0, 13.0, 1050.0, 53.0, 70.0, 24.0, 2.0, 48.0, 82.0, 141.0, 15.0, 38.0, 5298.0, 1564.0, 23.0, 2.0, 90.0, 1.0, 644.0, 65.0, 3761.0, 0.0, 9.0, 7.0, 0.0, 390.0, 2.0, 950.0, 8.0, 37.0, 940.0, 128.0, 210.0, 135.0, 10.0, 21.0, 29.0, 20.0, 3244.0, 641.0, 250.0, 7.0], "type": "choropleth"}], "name": "2020-04-19"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-04-26<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [16.0, 90.0, 1411.0, 5539.0, 14.0, 1229.0, 56.0, 65.0, 8.0, 2.0, 34.0, 67.0, 102.0, 9.0, 96.0, 3136.0, 1390.0, 18.0, 8.0, 83.0, 1.0, 592.0, 30.0, 2984.0, 7.0, 8.0, 15.0, 1.0, 665.0, 2.0, 794.0, 7.0, 36.0, 1702.0, 175.0, 189.0, 168.0, 6.0, 8.0, 33.0, 9.0, 2737.0, 654.0, 203.0, 0.0], "type": "choropleth"}], "name": "2020-04-26"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-05-03<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [12.0, 56.0, 750.0, 4239.0, 17.0, 1295.0, 71.0, 96.0, 24.0, 1.0, 28.0, 62.0, 988.0, 6.0, 40.0, 2068.0, 890.0, 10.0, 2.0, 68.0, 0.0, 493.0, 31.0, 2240.0, 4.0, 2.0, 8.0, 0.0, 803.0, 1.0, 581.0, 1.0, 10.0, 2516.0, 143.0, 140.0, 171.0, 6.0, 14.0, 44.0, 9.0, 2074.0, 485.0, 101.0, 4.0], "type": "choropleth"}], "name": "2020-05-03"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-05-10<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [2.0, 20.0, 812.0, 3175.0, 18.0, 1153.0, 52.0, 123.0, 11.0, 0.0, 32.0, 45.0, 563.0, 5.0, 37.0, 1485.0, 703.0, 7.0, 9.0, 73.0, 0.0, 437.0, 20.0, 1676.0, 2.0, 3.0, 5.0, 1.0, 1311.0, 1.0, 387.0, 1.0, 8.0, 3254.0, 122.0, 92.0, 171.0, 2.0, 6.0, 63.0, 4.0, 1357.0, 546.0, 59.0, 3.0], "type": "choropleth"}], "name": "2020-05-10"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-05-17<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [2.0, 11.0, 396.0, 2433.0, 17.0, 839.0, 138.0, 111.0, 5.0, 2.0, 18.0, 18.0, 609.0, 3.0, 31.0, 1730.0, 393.0, 12.0, 7.0, 38.0, 0.0, 348.0, 20.0, 1348.0, 1.0, 7.0, 6.0, 1.0, 1712.0, 0.0, 240.0, 0.0, 13.0, 3577.0, 125.0, 83.0, 146.0, 2.0, 2.0, 70.0, 7.0, 942.0, 454.0, 18.0, 0.0], "type": "choropleth"}], "name": "2020-05-17"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-05-24<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [3.0, 11.0, 228.0, 2139.0, 22.0, 665.0, 268.0, 153.0, 4.0, 0.0, 17.0, 15.0, 372.0, 1.0, 9.0, 259.0, 321.0, 8.0, 25.0, 35.0, 0.0, 429.0, 7.0, 877.0, 3.0, 5.0, 3.0, 0.0, 2217.0, 0.0, 142.0, 0.0, 3.0, 3809.0, 71.0, 98.0, 78.0, 0.0, 3.0, 165.0, 4.0, 1189.0, 319.0, 18.0, 3.0], "type": "choropleth"}], "name": "2020-05-24"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-05-31<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [1.0, 28.0, 187.0, 1488.0, 10.0, 580.0, 336.0, 212.0, 4.0, 0.0, 5.0, 12.0, 250.0, 4.0, 13.0, 433.0, 257.0, 4.0, 50.0, 40.0, 0.0, 380.0, 5.0, 630.0, 2.0, 6.0, 0.0, 3.0, 2536.0, 0.0, 134.0, 1.0, 1.0, 4066.0, 68.0, 94.0, 81.0, 0.0, 1.0, 254.0, 4.0, -1625.0, 397.0, 6.0, 0.0], "type": "choropleth"}], "name": "2020-05-31"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-06-07<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [-1.0, 4.0, 128.0, 1175.0, 20.0, 363.0, 583.0, 320.0, 1.0, 1.0, 7.0, 15.0, 263.0, 1.0, 3.0, 353.0, 145.0, 7.0, 144.0, 20.0, 0.0, 484.0, 13.0, 484.0, 1.0, 2.0, 0.0, 0.0, 3769.0, 0.0, 57.0, 0.0, 2.0, 4124.0, 93.0, 69.0, 67.0, 0.0, 1.0, 315.0, 2.0, 9.0, 264.0, 2.0, 1.0], "type": "choropleth"}], "name": "2020-06-07"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-06-14<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 5.0, 60.0, 749.0, 14.0, 220.0, 1686.0, 408.0, 3.0, 0.0, 2.0, 8.0, 275.0, 0.0, 3.0, 253.0, 116.0, 1.0, 132.0, 16.0, 0.0, 556.0, 5.0, 446.0, 3.0, 3.0, 0.0, 0.0, 3442.0, 0.0, 46.0, 0.0, 4.0, 4176.0, 90.0, 38.0, 77.0, 0.0, 0.0, 482.0, 4.0, 0.0, 215.0, 4.0, 0.0], "type": "choropleth"}], "name": "2020-06-14"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-06-21<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 13.0, 41.0, 512.0, 25.0, 137.0, 1156.0, 570.0, 0.0, 1.0, 7.0, 3.0, 327.0, 0.0, 0.0, 233.0, 94.0, 7.0, 147.0, 8.0, 0.0, 786.0, 3.0, 289.0, 2.0, 1.0, 0.0, 0.0, 4684.0, 0.0, 31.0, 0.0, 2.0, 4330.0, 109.0, 13.0, 102.0, 0.0, 0.0, 450.0, 3.0, 1187.0, 179.0, 4.0, 1.0], "type": "choropleth"}], "name": "2020-06-21"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-06-28<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [2.0, 12.0, 36.0, 441.0, 20.0, 123.0, 1030.0, 941.0, 0.0, 0.0, 12.0, 4.0, 206.0, 0.0, 2.0, 137.0, 73.0, 1.0, 196.0, 11.0, 0.0, 885.0, 18.0, 104.0, 0.0, 1.0, 0.0, 0.0, 4823.0, 2.0, 15.0, 0.0, 5.0, 3530.0, 82.0, 34.0, 100.0, 0.0, 2.0, 526.0, 2.0, 20.0, 227.0, 2.0, 0.0], "type": "choropleth"}], "name": "2020-06-28"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-07-05<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [2.0, 4.0, 39.0, 313.0, 27.0, 74.0, 799.0, 886.0, 6.0, 0.0, 0.0, 2.0, 352.0, 0.0, 1.0, 114.0, 55.0, 1.0, 220.0, 8.0, 0.0, 1063.0, 15.0, 123.0, 0.0, 1.0, 0.0, 0.0, 3991.0, 3.0, 22.0, 0.0, 2.0, 3755.0, 79.0, 50.0, 138.0, 0.0, 0.0, 743.0, 2.0, 42.0, 140.0, 2.0, 0.0], "type": "choropleth"}], "name": "2020-07-05"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-07-12<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [2.0, 2.0, 11.0, 213.0, 22.0, 64.0, 671.0, 1243.0, 6.0, 0.0, 4.0, 3.0, 266.0, 0.0, 0.0, 117.0, 48.0, 1.0, 272.0, 6.0, 0.0, 1258.0, 39.0, 93.0, 0.0, 0.0, 1.0, 0.0, 4367.0, 9.0, 10.0, 0.0, 1.0, 3903.0, 54.0, 46.0, 134.0, 0.0, 0.0, 880.0, 5.0, 18.0, 106.0, 3.0, 0.0], "type": "choropleth"}], "name": "2020-07-12"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-07-19<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [15.0, 3.0, 18.0, 150.0, 32.0, 56.0, 1524.0, 1429.0, 1.0, 0.0, 7.0, 2.0, 266.0, 0.0, -1.0, 146.0, 21.0, 1.0, 266.0, 1.0, 0.0, 1359.0, 56.0, 91.0, 1.0, 1.0, 0.0, 0.0, 4178.0, 9.0, -1.0, 0.0, 3.0, 4901.0, 53.0, 29.0, 142.0, 0.0, 1.0, 954.0, 7.0, 17.0, 93.0, 2.0, 0.0], "type": "choropleth"}], "name": "2020-07-19"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-07-26<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [38.0, 1.0, 21.0, 116.0, 40.0, 35.0, 609.0, 1789.0, 16.0, 0.0, 12.0, 2.0, 202.0, 0.0, 1.0, 38.0, 32.0, 8.0, 249.0, 0.0, 0.0, 1512.0, 69.0, 62.0, 0.0, 0.0, 1.0, 0.0, 4496.0, 11.0, 23.0, 0.0, 0.0, 3480.0, 47.0, 28.0, 161.0, 0.0, 4.0, 1736.0, 3.0, 12.0, 78.0, 8.0, 0.0], "type": "choropleth"}], "name": "2020-07-26"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-08-02<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [60.0, 6.0, 24.0, 96.0, 48.0, 54.0, 496.0, 2125.0, 13.0, 0.0, 13.0, 2.0, 221.0, -6.0, 0.0, 72.0, 30.0, 6.0, 261.0, 1.0, 0.0, 1490.0, 84.0, 47.0, 1.0, 0.0, 5.0, 0.0, 4066.0, 8.0, -9.0, 0.0, 0.0, 4376.0, 60.0, 21.0, 226.0, 1.0, 4.0, 1597.0, 2.0, 13.0, 46.0, 5.0, 1.0], "type": "choropleth"}], "name": "2020-08-02"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-08-09<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [92.0, 3.0, 27.0, 71.0, 59.0, 35.0, 469.0, 2192.0, 8.0, 0.0, 6.0, 2.0, 186.0, 0.0, 2.0, 61.0, 48.0, 4.0, 216.0, 5.0, 0.0, 1237.0, 77.0, 51.0, 0.0, 1.0, 3.0, 0.0, 4552.0, 13.0, 9.0, 0.0, 1.0, 4433.0, 76.0, 18.0, 287.0, 2.0, 7.0, 2042.0, 4.0, 58.0, 20.0, 5.0, 0.0], "type": "choropleth"}], "name": "2020-08-09"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-08-16<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [108.0, 7.0, 67.0, 88.0, 51.0, 39.0, 375.0, 2255.0, 9.0, 1.0, 7.0, 4.0, 148.0, 0.0, 2.0, 86.0, 33.0, 16.0, 168.0, 6.0, 0.0, 1212.0, 98.0, 191.0, 0.0, 0.0, 3.0, 0.0, 4459.0, 13.0, 16.0, 0.0, 5.0, 4322.0, 70.0, 22.0, 291.0, 0.0, 2.0, 1431.0, 0.0, 114.0, 20.0, 5.0, 3.0], "type": "choropleth"}], "name": "2020-08-16"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-08-23<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [96.0, 4.0, 53.0, 64.0, 47.0, 40.0, 400.0, 2219.0, 5.0, 1.0, 15.0, 1.0, 240.0, 0.0, 1.0, 103.0, 40.0, 14.0, 215.0, 5.0, 0.0, 1004.0, 78.0, 41.0, 1.0, 3.0, 1.0, 1.0, 3723.0, 7.0, 31.0, 0.0, 3.0, 3786.0, 78.0, 18.0, 281.0, 2.0, 2.0, 1220.0, 4.0, 221.0, 27.0, 5.0, 17.0], "type": "choropleth"}], "name": "2020-08-23"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-08-30<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [135.0, 1.0, -98.0, 71.0, 68.0, 39.0, 392.0, 2047.0, 13.0, 0.0, 11.0, 2.0, 245.0, 1.0, 1.0, 97.0, 25.0, 20.0, 146.0, 1.0, 0.0, 819.0, 88.0, 40.0, 1.0, 2.0, 0.0, 2.0, 3678.0, 14.0, 27.0, 0.0, 0.0, 3401.0, 78.0, 23.0, 306.0, 0.0, 2.0, 969.0, 15.0, 173.0, 11.0, 4.0, 5.0], "type": "choropleth"}], "name": "2020-08-30"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-09-06<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [110.0, 3.0, 13.0, 54.0, 63.0, 24.0, 348.0, 2049.0, 14.0, 0.0, 13.0, 3.0, 169.0, 0.0, 1.0, 97.0, 30.0, 22.0, 112.0, 10.0, 0.0, 831.0, 107.0, 64.0, 1.0, 0.0, 0.0, 2.0, 3400.0, 10.0, 25.0, 2.0, 0.0, 2674.0, 87.0, 21.0, 315.0, 4.0, 2.0, 861.0, 12.0, 407.0, 14.0, 10.0, 17.0], "type": "choropleth"}], "name": "2020-09-06"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-09-13<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [54.0, 20.0, 18.0, 77.0, 44.0, 27.0, 357.0, 1512.0, 26.0, 0.0, 20.0, 4.0, 327.0, 0.0, 1.0, 217.0, 24.0, 21.0, 105.0, 13.0, 0.0, 864.0, 118.0, 69.0, 0.0, 2.0, 0.0, 1.0, 3263.0, 12.0, 15.0, 0.0, 1.0, 2214.0, 68.0, 27.0, 270.0, 1.0, 0.0, 558.0, 27.0, 329.0, 11.0, 13.0, 14.0], "type": "choropleth"}], "name": "2020-09-13"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-09-20<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [35.0, 10.0, 23.0, 149.0, 41.0, 52.0, 337.0, 1284.0, 24.0, 1.0, 47.0, 7.0, 187.0, 0.0, 2.0, 369.0, 36.0, 33.0, 162.0, 46.0, 0.0, 1144.0, 148.0, 97.0, 1.0, 0.0, 0.0, 5.0, 2672.0, 16.0, 32.0, 1.0, 2.0, 1889.0, 105.0, 45.0, 272.0, 1.0, 7.0, 506.0, 22.0, 748.0, 19.0, 20.0, 52.0], "type": "choropleth"}], "name": "2020-09-20"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-09-27<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [24.0, 21.0, 32.0, 211.0, 35.0, 55.0, 355.0, 1280.0, 24.0, 0.0, 103.0, 11.0, 189.0, 0.0, 4.0, 443.0, 74.0, 41.0, 110.0, 53.0, 0.0, 1288.0, 200.0, 128.0, 0.0, 5.0, 0.0, 11.0, 2937.0, 22.0, 99.0, 0.0, 3.0, 1591.0, 139.0, 41.0, 283.0, 5.0, 5.0, 445.0, 21.0, 737.0, 15.0, 14.0, 55.0], "type": "choropleth"}], "name": "2020-09-27"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-10-04<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [19.0, 26.0, 84.0, 363.0, 48.0, 227.0, 338.0, 1224.0, 26.0, 0.0, 121.0, 9.0, 368.0, 3.0, 2.0, 487.0, 69.0, 30.0, 64.0, 86.0, 0.0, 1368.0, 245.0, 151.0, 2.0, 5.0, 1.0, 8.0, 2658.0, 21.0, 85.0, 0.0, 5.0, 1213.0, 198.0, 52.0, 285.0, 11.0, 8.0, 578.0, 16.0, 854.0, 15.0, 10.0, 107.0], "type": "choropleth"}], "name": "2020-10-04"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-10-11<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [4.0, 38.0, 127.0, 475.0, 48.0, 138.0, 339.0, 1122.0, 26.0, 3.0, 260.0, 11.0, 544.0, 1.0, 1.0, 502.0, 93.0, 40.0, 91.0, 132.0, 0.0, 1587.0, 266.0, 180.0, 2.0, 8.0, 7.0, 2.0, 4693.0, 23.0, 134.0, 0.0, 0.0, 1170.0, 374.0, 75.0, 408.0, 6.0, 12.0, 804.0, 11.0, 843.0, -1.0, 16.0, 157.0], "type": "choropleth"}], "name": "2020-10-11"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-10-18<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [7.0, 42.0, 222.0, 821.0, 94.0, 152.0, 317.0, 1136.0, 39.0, 0.0, 435.0, 11.0, 196.0, 0.0, 5.0, 753.0, 172.0, 60.0, 146.0, 188.0, 1.0, 1831.0, 233.0, 377.0, 4.0, 11.0, 1.0, 4.0, 2386.0, 34.0, 169.0, 0.0, 3.0, 978.0, 569.0, 101.0, 461.0, 27.0, 21.0, 691.0, 11.0, 846.0, 24.0, 51.0, 148.0], "type": "choropleth"}], "name": "2020-10-18"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-10-25<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 86.0, 397.0, 1250.0, 108.0, 197.0, 309.0, 1184.0, 74.0, 0.0, 779.0, 22.0, 166.0, 5.0, 2.0, 1166.0, 264.0, 65.0, 114.0, 283.0, 0.0, 2241.0, 193.0, 795.0, 12.0, 22.0, 8.0, 6.0, 2757.0, 30.0, 297.0, 0.0, 1.0, 860.0, 865.0, 135.0, 519.0, 71.0, 52.0, 497.0, 13.0, 977.0, 15.0, 140.0, 193.0], "type": "choropleth"}], "name": "2020-10-25"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-11-01<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [2.0, 151.0, 927.0, 1821.0, 204.0, 256.0, 303.0, 1361.0, 125.0, 1.0, 1228.0, 21.0, 131.0, 0.0, 5.0, 2409.0, 451.0, 61.0, 94.0, 394.0, 1.0, 2682.0, 140.0, 1488.0, 18.0, 33.0, 11.0, 13.0, 2971.0, 47.0, 390.0, 0.0, 3.0, 772.0, 1345.0, 228.0, 676.0, 60.0, 123.0, 443.0, 11.0, 1126.0, 5.0, 316.0, 529.0], "type": "choropleth"}], "name": "2020-11-01"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-11-08<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 281.0, 1318.0, 2327.0, 367.0, 328.0, 296.0, 1276.0, 232.0, 1.0, 1429.0, 20.0, 146.0, 2.0, 4.0, 3432.0, 859.0, 149.0, 83.0, 619.0, 8.0, 2993.0, 103.0, 2568.0, 25.0, 49.0, 28.0, 12.0, 3132.0, 24.0, 574.0, 0.0, 3.0, 739.0, 2089.0, 352.0, 812.0, 132.0, 191.0, 398.0, 12.0, 2955.0, 84.0, 557.0, 525.0], "type": "choropleth"}], "name": "2020-11-08"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-11-15<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 418.0, 1366.0, 2892.0, 465.0, 437.0, 276.0, 1240.0, 255.0, 12.0, 1350.0, 17.0, 178.0, 6.0, 7.0, 4123.0, 1201.0, 322.0, 111.0, 659.0, 5.0, 3202.0, 61.0, 3835.0, 24.0, 91.0, 39.0, 21.0, 3515.0, 54.0, 487.0, 0.0, 9.0, 730.0, 2476.0, 485.0, 1047.0, 159.0, 243.0, 432.0, 14.0, 1936.0, 142.0, 664.0, 472.0], "type": "choropleth"}], "name": "2020-11-15"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-11-22<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 559.0, 1197.0, 3094.0, 750.0, 505.0, 250.0, 1256.0, 304.0, 4.0, 988.0, 24.0, 193.0, 7.0, 6.0, 4192.0, 1586.0, 524.0, 144.0, 703.0, 1.0, 3309.0, 69.0, 4594.0, 30.0, 144.0, 41.0, 14.0, 3134.0, 48.0, 408.0, 0.0, 12.0, 646.0, 3270.0, 516.0, 1121.0, 161.0, 255.0, 662.0, 15.0, 1850.0, 242.0, 663.0, 476.0], "type": "choropleth"}], "name": "2020-11-22"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-11-29<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [1.0, 717.0, 929.0, 3222.0, 934.0, 547.0, 287.0, 1297.0, 359.0, 5.0, 942.0, 45.0, 222.0, 24.0, 18.0, 3605.0, 2147.0, 691.0, 90.0, 872.0, 0.0, 3072.0, 51.0, 5081.0, 44.0, 154.0, 40.0, 22.0, 3979.0, 48.0, 459.0, 0.0, 22.0, 661.0, 3411.0, 530.0, 1146.0, 145.0, 332.0, 574.0, 17.0, 2049.0, 275.0, 625.0, 398.0], "type": "choropleth"}], "name": "2020-11-29"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-12-06<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 735.0, 773.0, 3000.0, 983.0, 645.0, 272.0, 1224.0, 462.0, 11.0, 764.0, 56.0, 355.0, 19.0, 22.0, 2836.0, 2683.0, 682.0, 84.0, 1196.0, 1.0, 2436.0, 53.0, 5174.0, 65.0, 197.0, 45.0, 18.0, 4062.0, 46.0, 342.0, 0.0, 26.0, 686.0, 3060.0, 536.0, 1127.0, 180.0, 360.0, 729.0, 23.0, 1584.0, 386.0, 638.0, 342.0], "type": "choropleth"}], "name": "2020-12-06"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-12-13<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 633.0, 631.0, 2925.0, 891.0, 762.0, 258.0, 1245.0, 466.0, 19.0, 633.0, 56.0, 97.0, 18.0, 38.0, 2768.0, 3117.0, 622.0, 173.0, 1097.0, 1.0, 1886.0, 82.0, 4442.0, 87.0, 249.0, 47.0, 18.0, 4236.0, 49.0, 366.0, 0.0, 33.0, 747.0, 2775.0, 596.0, 1065.0, 179.0, 319.0, 1070.0, 38.0, 1372.0, 447.0, 651.0, 333.0], "type": "choropleth"}], "name": "2020-12-13"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-12-20<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 878.0, 675.0, 3236.0, 921.0, 837.0, 268.0, 1422.0, 537.0, 12.0, 796.0, 94.0, 73.0, 25.0, 36.0, 2650.0, 4294.0, 547.0, 233.0, 1134.0, 0.0, 1429.0, 100.0, 4279.0, 90.0, 254.0, 42.0, 21.0, 4249.0, 44.0, 444.0, 0.0, 17.0, 816.0, 2533.0, 575.0, 1009.0, 380.0, 290.0, 1415.0, 111.0, 1302.0, 479.0, 677.0, 264.0], "type": "choropleth"}], "name": "2020-12-20"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-12-27<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [1.0, 530.0, 574.0, 3357.0, 555.0, 797.0, 289.0, 1696.0, 494.0, 22.0, 713.0, 139.0, 44.0, 30.0, 35.0, 2202.0, 3897.0, 434.0, 112.0, 948.0, 0.0, 1068.0, 127.0, 3126.0, 120.0, 320.0, 24.0, 20.0, 4224.0, 40.0, 512.0, 0.0, 17.0, 880.0, 1721.0, 485.0, 836.0, 218.0, 212.0, 2044.0, 121.0, 898.0, 286.0, 624.0, 308.0], "type": "choropleth"}], "name": "2020-12-27"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-01-03<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 443.0, 501.0, 4277.0, 514.0, 835.0, 324.0, 1794.0, 401.0, 18.0, 916.0, 200.0, 67.0, 40.0, 37.0, 2297.0, 4494.0, 351.0, 65.0, 837.0, 1.0, 847.0, 190.0, 3407.0, 121.0, 325.0, 37.0, 10.0, 4787.0, 23.0, 617.0, 0.0, 15.0, 1082.0, 2001.0, 499.0, 749.0, 544.0, 238.0, 2842.0, 162.0, 1013.0, 448.0, 549.0], "type": "choropleth"}], "name": "2021-01-03"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-01-10<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 399.0, 377.0, 6430.0, 448.0, 1068.0, 329.0, 2149.0, 296.0, 18.0, 1155.0, 197.0, 118.0, 39.0, 25.0, 2721.0, 6145.0, 306.0, 192.0, 764.0, 0.0, 631.0, 255.0, 3423.0, 169.0, 265.0, 32.0, 13.0, 6493.0, 36.0, 754.0, 0.0, 36.0, 1401.0, 2070.0, 685.0, 675.0, 601.0, 195.0, 3586.0, 159.0, 1037.0, 706.0, 439.0, 415.0], "type": "choropleth"}], "name": "2021-01-10"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-01-17<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 359.0, 357.0, 7862.0, 357.0, 1032.0, 381.0, 2517.0, 248.0, 20.0, 1223.0, 205.0, 142.0, 42.0, 32.0, 2537.0, 5965.0, 206.0, 229.0, 693.0, 0.0, 632.0, 334.0, 3422.0, 129.0, 247.0, 22.0, 6.0, 6998.0, 20.0, 646.0, 0.0, 45.0, 2027.0, 2166.0, 1058.0, 567.0, 556.0, 182.0, 3942.0, 124.0, 1440.0, 890.0, 396.0, 477.0], "type": "choropleth"}], "name": "2021-01-17"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-01-24<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 336.0, 344.0, 8700.0, 337.0, 1013.0, 456.0, 2743.0, 211.0, 16.0, 1031.0, 208.0, 304.0, 51.0, 26.0, 2768.0, 5395.0, 177.0, 211.0, 627.0, 0.0, 580.0, 414.0, 3284.0, 133.0, 204.0, 13.0, 12.0, 8910.0, 23.0, 539.0, 0.0, 27.0, 2907.0, 2008.0, 1608.0, 555.0, 594.0, 180.0, 3769.0, 96.0, 2127.0, 682.0, 352.0, 542.0], "type": "choropleth"}], "name": "2021-01-24"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-01-31<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 303.0, 313.0, 8238.0, 225.0, 922.0, 519.0, 2609.0, 200.0, 14.0, 939.0, 142.0, 236.0, 43.0, 27.0, 3011.0, 4867.0, 150.0, 178.0, 556.0, 0.0, 576.0, 377.0, 3055.0, 84.0, 154.0, 15.0, 16.0, 8922.0, 37.0, 462.0, 0.0, 20.0, 3900.0, 1817.0, 2013.0, 559.0, 574.0, 143.0, 3290.0, 65.0, 2878.0, 586.0, 251.0, 520.0], "type": "choropleth"}], "name": "2021-01-31"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-02-07<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 291.0, 297.0, 6314.0, 286.0, 719.0, 522.0, 2010.0, 142.0, 14.0, 927.0, 91.0, 153.0, 42.0, 17.0, 2910.0, 4545.0, 176.0, 241.0, 566.0, 0.0, 510.0, 325.0, 2757.0, 144.0, 129.0, 13.0, 15.0, 7664.0, 33.0, 408.0, 0.0, 18.0, 4052.0, 1907.0, 1676.0, 546.0, 557.0, 120.0, 2126.0, 49.0, 3067.0, 524.0, 193.0, 460.0], "type": "choropleth"}], "name": "2021-02-07"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-02-14<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Tunisia"], "name": "", "z": [0.0, 199.0, 273.0, 4706.0, 293.0, 501.0, 567.0, 1612.0, 149.0, 9.0, 908.0, 76.0, 309.0, 34.0, 22.0, 2851.0, 3399.0, 154.0, 240.0, 616.0, 0.0, 476.0, 267.0, 2304.0, 112.0, 133.0, 16.0, 11.0, 8007.0, 52.0, 420.0, 0.0, 10.0, 4761.0, 1720.0, 1163.0, 485.0, 686.0, 94.0, 1609.0, 53.0, 3361.0, 313.0, 137.0, 330.0], "type": "choropleth"}], "name": "2021-02-14"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-02-21<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 175.0, 241.0, 3423.0, 230.0, 414.0, 501.0, 1229.0, 111.0, 9.0, 1071.0, 46.0, 215.0, 40.0, 16.0, 2468.0, 2839.0, 171.0, 158.0, 593.0, 0.0, 538.0, 189.0, 2141.0, 91.0, 101.0, 16.0, 11.0, 5900.0, 60.0, 404.0, 1.0, 15.0, 5122.0, 1364.0, 641.0, 481.0, 620.0, 59.0, 1154.0, 35.0, 2354.0, 221.0, 93.0], "type": "choropleth"}], "name": "2021-02-21"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-02-28<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 175.0, 174.0, 2273.0, 337.0, 320.0, 530.0, 932.0, 97.0, 2.0, 1125.0, 23.0, 275.0, 54.0, 16.0, 2150.0, 2206.0, 207.0, 111.0, 675.0, 0.0, 590.0, 175.0, 1981.0, 76.0, 73.0, 15.0, 11.0, 5608.0, 53.0, 348.0, 0.0, 15.0, 4757.0, 1598.0, 355.0, 503.0, 684.0, 60.0, 940.0, 43.0, 2041.0, 177.0, 70.0], "type": "choropleth"}], "name": "2021-02-28"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-03-07<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 152.0, 184.0, 1653.0, 423.0, 245.0, 505.0, 737.0, 64.0, 1.0, 1378.0, 18.0, 228.0, 64.0, 25.0, 2147.0, 1832.0, 254.0, 80.0, 899.0, 0.0, 614.0, 139.0, 2086.0, 77.0, 84.0, 15.0, 19.0, 4889.0, 64.0, 274.0, 0.0, 10.0, 4737.0, 1516.0, 223.0, 550.0, 647.0, 55.0, 685.0, 37.0, 1996.0, 177.0, 58.0], "type": "choropleth"}], "name": "2021-03-07"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-03-14<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 160.0, 180.0, 1017.0, 671.0, 224.0, 597.0, 640.0, 87.0, 6.0, 1509.0, 12.0, 197.0, 66.0, 19.0, 1857.0, 1479.0, 333.0, 95.0, 1079.0, 0.0, 543.0, 120.0, 2360.0, 72.0, 68.0, 29.0, 17.0, 4106.0, 62.0, 238.0, 0.0, 7.0, 4296.0, 1893.0, 144.0, 583.0, 692.0, 43.0, 648.0, 33.0, 1120.0, 143.0, 69.0], "type": "choropleth"}], "name": "2021-03-14"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-03-21<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 201.0, 266.0, 640.0, 734.0, 213.0, 605.0, 885.0, 96.0, 4.0, 1441.0, 9.0, 215.0, 68.0, 19.0, 1850.0, 1293.0, 371.0, 132.0, 1310.0, 0.0, 567.0, 81.0, 2797.0, 59.0, 80.0, 33.0, 21.0, 3326.0, 70.0, 211.0, 0.0, 9.0, 4741.0, 2122.0, 84.0, 725.0, 516.0, 38.0, 785.0, 22.0, 652.0, 116.0, 59.0], "type": "choropleth"}], "name": "2021-03-21"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-03-28<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 182.0, 190.0, 441.0, 691.0, 204.0, 475.0, 927.0, 120.0, 8.0, 1207.0, 14.0, 287.0, 81.0, 12.0, 2321.0, 1203.0, 418.0, 102.0, 1710.0, 0.0, 600.0, 93.0, 2991.0, 52.0, 75.0, 24.0, 15.0, 3587.0, 46.0, 196.0, 0.0, 8.0, 4994.0, 2584.0, 69.0, 906.0, 452.0, 46.0, 552.0, 29.0, 2100.0, 140.0, 62.0], "type": "choropleth"}], "name": "2021-03-28"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-04-04<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 193.0, 272.0, 244.0, 879.0, 194.0, 890.0, 1139.0, 165.0, 11.0, 1071.0, 15.0, 218.0, 73.0, 29.0, 2055.0, 1101.0, 500.0, 89.0, 1743.0, 0.0, 763.0, 58.0, 3097.0, 53.0, 59.0, 12.0, 11.0, 2524.0, 60.0, 183.0, 0.0, 17.0, 5157.0, 3057.0, 42.0, 956.0, 529.0, 57.0, 324.0, 22.0, 688.0, 96.0, 67.0], "type": "choropleth"}], "name": "2021-04-04"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-04-11<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 238.0, 304.0, 253.0, 829.0, 241.0, 702.0, 1795.0, 250.0, 11.0, 863.0, 12.0, 337.0, 87.0, 22.0, 2101.0, 1440.0, 505.0, 125.0, 1702.0, 0.0, 1330.0, 53.0, 3224.0, 55.0, 77.0, 13.0, 4.0, 5191.0, 68.0, 169.0, 0.0, 11.0, 5460.0, 3480.0, 37.0, 1065.0, 540.0, 44.0, 335.0, 22.0, 630.0, 123.0, 70.0], "type": "choropleth"}], "name": "2021-04-11"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-04-18<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [1.0, 211.0, 274.0, 187.0, 777.0, 308.0, 831.0, 2439.0, 254.0, 16.0, 618.0, 14.0, 410.0, 64.0, 19.0, 1983.0, 1552.0, 577.0, 205.0, 1767.0, 0.0, 2242.0, 39.0, 2673.0, 67.0, 73.0, 22.0, 7.0, 3001.0, 56.0, 168.0, 0.0, 24.0, 5356.0, 3611.0, 29.0, 1097.0, 541.0, 40.0, 414.0, 31.0, 653.0, 167.0, 45.0], "type": "choropleth"}], "name": "2021-04-18"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-04-25<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 189.0, 277.0, 163.0, 712.0, 342.0, 679.0, 3023.0, 292.0, 10.0, 536.0, 20.0, 474.0, 51.0, 16.0, 2125.0, 1619.0, 545.0, 189.0, 1441.0, 0.0, 2842.0, 17.0, 2311.0, 49.0, 88.0, 5.0, 4.0, 2608.0, 36.0, 154.0, 0.0, 28.0, 5963.0, 3383.0, 20.0, 1162.0, 389.0, 49.0, 412.0, 16.0, 610.0, 135.0, 58.0], "type": "choropleth"}], "name": "2021-04-25"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-05-02<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 158.0, 267.0, 115.0, 585.0, 335.0, 705.0, 3126.0, 328.0, 15.0, 381.0, 15.0, 563.0, 25.0, 11.0, 1964.0, 1621.0, 446.0, 148.0, 1177.0, 0.0, 2910.0, 14.0, 1939.0, 42.0, 102.0, 6.0, 3.0, 2286.0, 40.0, 144.0, 0.0, 20.0, 4914.0, 2653.0, 12.0, 888.0, 271.0, 55.0, 269.0, 17.0, 625.0, 125.0, 67.0], "type": "choropleth"}], "name": "2021-05-02"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-05-09<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 137.0, 260.0, 69.0, 437.0, 326.0, 657.0, 3377.0, 287.0, 21.0, 324.0, 7.0, 482.0, 33.0, 8.0, 1573.0, 1552.0, 576.0, 175.0, 800.0, 0.0, 2426.0, 11.0, 1656.0, 68.0, 84.0, 5.0, 1.0, 1752.0, 35.0, 157.0, 0.0, 11.0, 4983.0, 1944.0, 15.0, 684.0, 253.0, 30.0, 318.0, 41.0, 576.0, 125.0, 47.0], "type": "choropleth"}], "name": "2021-05-09"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-05-16<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 92.0, 158.0, 76.0, 330.0, 322.0, 614.0, 3446.0, 239.0, 12.0, 234.0, 5.0, 477.0, 21.0, 9.0, 1224.0, 1322.0, 386.0, 154.0, 573.0, 0.0, 2026.0, 5.0, 1323.0, 54.0, 69.0, 5.0, 0.0, 1452.0, 21.0, 135.0, 0.0, 7.0, 4093.0, 1652.0, 15.0, 557.0, 205.0, 31.0, 475.0, 28.0, 547.0, 102.0, 29.0], "type": "choropleth"}], "name": "2021-05-16"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-05-23<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Montenegro", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 72.0, 133.0, 42.0, 237.0, 283.0, 686.0, 3424.0, 195.0, 7.0, 119.0, 6.0, 494.0, 18.0, 1.0, 981.0, 1263.0, 357.0, 113.0, 344.0, 0.0, 1661.0, 22.0, 1069.0, 60.0, 82.0, 4.0, 0.0, 1210.0, 13.0, 100.0, 0.0, 7.0, 3409.0, 1264.0, 10.0, 418.0, 68.0, 29.0, 592.0, 31.0, 281.0, 91.0, 37.0], "type": "choropleth"}], "name": "2021-05-23"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-05-30<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 50.0, 98.0, 60.0, 166.0, 281.0, 650.0, 3558.0, 111.0, 7.0, 84.0, 8.0, 352.0, 11.0, 16.0, 807.0, 1002.0, 282.0, 137.0, 209.0, 1.0, 1342.0, 7.0, 821.0, 44.0, 72.0, 4.0, 2.0, 1860.0, 78.0, 0.0, 2.0, 2786.0, 810.0, 6.0, 335.0, 47.0, 21.0, 561.0, 25.0, 285.0, 85.0, 26.0], "type": "choropleth"}], "name": "2021-05-30"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-06-06<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 43.0, 93.0, 60.0, 158.0, 212.0, 769.0, 3679.0, 72.0, 3.0, 55.0, 2.0, 264.0, 12.0, 11.0, 603.0, 818.0, 223.0, 157.0, 126.0, 0.0, 1124.0, 7.0, 477.0, 43.0, 50.0, 4.0, 0.0, 1025.0, 59.0, 0.0, 2.0, 2682.0, 414.0, 11.0, 539.0, 65.0, 14.0, 611.0, 15.0, 291.0, 72.0, 22.0], "type": "choropleth"}], "name": "2021-06-06"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-06-13<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 22.0, 55.0, 65.0, 80.0, 207.0, 770.0, 3817.0, 46.0, 10.0, 66.0, 7.0, 239.0, 3.0, 5.0, 396.0, 600.0, 145.0, 161.0, 50.0, 0.0, 1035.0, 12.0, 479.0, 43.0, 32.0, 0.0, 0.0, 1346.0, 35.0, 0.0, 4.0, 2635.0, 421.0, 13.0, 1019.0, 35.0, 16.0, 791.0, 14.0, 305.0, 51.0, 17.0], "type": "choropleth"}], "name": "2021-06-13"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-06-20<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 19.0, 48.0, 72.0, 90.0, 145.0, 806.0, 4156.0, 42.0, 1.0, 55.0, 5.0, 245.0, 2.0, 3.0, 348.0, 551.0, 126.0, 239.0, 46.0, 0.0, 867.0, -3.0, 268.0, 40.0, 27.0, 0.0, 1.0, 1037.0, 18.0, 0.0, 1.0, 1494.0, 255.0, 18.0, 492.0, 39.0, 8.0, 937.0, 16.0, 151.0, 0.0, 15.0], "type": "choropleth"}], "name": "2021-06-20"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-06-27<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 20.0, 24.0, 124.0, 37.0, 151.0, 785.0, 4744.0, 26.0, 0.0, 18.0, 2.0, 230.0, 1.0, 2.0, 231.0, 368.0, 116.0, 356.0, 30.0, 0.0, 880.0, 2.0, 202.0, 7.0, 14.0, 0.0, 0.0, 1377.0, 17.0, 0.0, 2.0, 1382.0, 151.0, 19.0, 927.0, 27.0, 7.0, 1198.0, 11.0, 127.0, 45.0, 9.0], "type": "choropleth"}], "name": "2021-06-27"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-07-04<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [0.0, 9.0, 25.0, 122.0, 57.0, 133.0, 805.0, 4218.0, 18.0, 0.0, 12.0, 5.0, 137.0, 0.0, 4.0, 195.0, 271.0, 73.0, 411.0, 12.0, 0.0, 947.0, -1.0, 177.0, 25.0, 10.0, 0.0, 0.0, 1058.0, 15.0, 0.0, 2.0, 1646.0, 105.0, 28.0, 675.0, 8.0, 0.0, 1940.0, 13.0, 132.0, 12.0, 1.0], "type": "choropleth"}], "name": "2021-07-04"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-07-11<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Canada", "Chile", "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [1.0, 14.0, 13.0, 205.0, 56.0, 76.0, 774.0, 3930.0, 9.0, 6.0, 21.0, 2.0, 170.0, 1.0, 3.0, 167.0, 202.0, 55.0, 252.0, 12.0, 0.0, 1067.0, 10.0, 126.0, 8.0, 8.0, 1.0, 0.0, 1347.0, 10.0, 0.0, 2.0, 1157.0, 76.0, 44.0, 289.0, 6.0, 4.0, 2449.0, 16.0, 92.0, 12.0, 2.0], "type": "choropleth"}], "name": "2021-07-11"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-07-18<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Chile", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [3.0, 6.0, 11.0, 297.0, 34.0, 637.0, 11.0, 4.0, 7.0, 3.0, 123.0, 1.0, 3.0, 147.0, 129.0, 58.0, 201.0, 11.0, 0.0, 1302.0, 11.0, 92.0, 7.0, 6.0, 1.0, 0.0, 1362.0, 12.0, 0.0, 0.0, 759.0, 55.0, 51.0, 36.0, 5.0, 2.0, 2570.0, 14.0, 93.0, 3.0, 2.0], "type": "choropleth"}], "name": "2021-07-18"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-07-25<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Australia", "Austria", "Belgium", "Britain", "Bulgaria", "Chile", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [4.0, 2.0, 11.0, 458.0, 20.0, 512.0, 7.0, 15.0, 19.0, 0.0, 58.0, 0.0, 1.0, 152.0, 164.0, 48.0, 186.0, 5.0, 0.0, 1639.0, 11.0, 82.0, 7.0, 6.0, 1.0, 1.0, 2093.0, 25.0, 0.0, 3.0, 744.0, 27.0, 85.0, 15.0, 10.0, 3.0, 2916.0, 19.0, 125.0, 5.0, 4.0], "type": "choropleth"}], "name": "2021-07-25"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-08-01<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Austria", "Belgium", "Britain", "Bulgaria", "Chile", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [7.0, 21.0, 568.0, 21.0, 502.0, 18.0, 22.0, 17.0, 8.0, 837.0, 1.0, 5.0, 273.0, 132.0, 77.0, 313.0, 6.0, 0.0, 2196.0, 17.0, 119.0, 6.0, 6.0, 1.0, 2.0, 2610.0, 31.0, 0.0, 0.0, 548.0, 19.0, 77.0, 18.0, 6.0, 1.0, 2416.0, 22.0, 265.0, 4.0, 12.0], "type": "choropleth"}], "name": "2021-08-01"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-08-08<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Austria", "Belgium", "Britain", "Bulgaria", "Chile", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [12.0, 27.0, 616.0, 40.0, 488.0, 9.0, 18.0, -11.0, 3.0, 154.0, 5.0, 6.0, 348.0, 125.0, 100.0, 358.0, 7.0, 0.0, 3019.0, 65.0, 152.0, 3.0, 11.0, 2.0, 3.0, 3386.0, 46.0, 0.0, 5.0, 512.0, 24.0, 98.0, 30.0, 1.0, 4.0, 2622.0, 26.0, 520.0, 2.0, 4.0], "type": "choropleth"}], "name": "2021-08-08"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-08-15<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Austria", "Belgium", "Britain", "Bulgaria", "Chile", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iceland", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [6.0, 19.0, 639.0, 89.0, 364.0, 11.0, 22.0, 10.0, 8.0, 82.0, 2.0, 12.0, 498.0, 87.0, 148.0, 384.0, 5.0, 0.0, 3813.0, 126.0, 212.0, 2.0, 24.0, 4.0, 4.0, 3747.0, 55.0, 0.0, 4.0, 443.0, 14.0, 95.0, 32.0, 3.0, 0.0, 2328.0, 42.0, 464.0, 1.0, 15.0], "type": "choropleth"}], "name": "2021-08-15"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-08-22<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Austria", "Belgium", "Britain", "Bulgaria", "Chile", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [6.0, 33.0, 689.0, 131.0, 270.0, 18.0, 23.0, 12.0, 6.0, 115.0, 2.0, 11.0, 850.0, 109.0, 161.0, 345.0, 8.0, 4210.0, 162.0, 319.0, 8.0, 36.0, 2.0, 6.0, 4988.0, 57.0, 0.0, 3.0, 486.0, 17.0, 77.0, 64.0, 3.0, 7.0, 2280.0, 55.0, 666.0, 10.0, 28.0], "type": "choropleth"}], "name": "2021-08-22"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-08-29<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Austria", "Belgium", "Britain", "Bulgaria", "Chile", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iran", "Israel", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland"], "name": "", "z": [19.0, 40.0, 802.0, 256.0, 235.0, 24.0, 21.0, 17.0, 14.0, 247.0, 8.0, 10.0, 852.0, 159.0, 252.0, 358.0, 11.0, 4444.0, 159.0, 342.0, 4.0, 52.0, 0.0, 4.0, 5010.0, 67.0, 0.0, 3.0, 236.0, 24.0, 82.0, 116.0, 1.0, 8.0, 2174.0, 62.0, 864.0, 17.0, 42.0], "type": "choropleth"}], "name": "2021-08-29"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-09-05<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Austria", "Belgium", "Britain", "Bulgaria", "Chile", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iran", "Israel", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "Spain", "Sweden", "Switzerland"], "name": "", "z": [12.0, 37.0, 793.0, 384.0, 205.0, 45.0, 16.0, 4.0, 10.0, 119.0, 12.0, 15.0, 774.0, 214.0, 250.0, 407.0, 4.0, 4192.0, 216.0, 12.0, 68.0, 0.0, 4.0, 4975.0, 68.0, 1.0, 8.0, 373.0, 39.0, 77.0, 161.0, 1.0, 7.0, 1824.0, 795.0, 7.0, 44.0], "type": "choropleth"}], "name": "2021-09-05"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-09-12<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Austria", "Belgium", "Britain", "Bulgaria", "Chile", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iran", "Israel", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "Spain", "Sweden", "Switzerland"], "name": "", "z": [46.0, 57.0, 972.0, 407.0, 142.0, 70.0, 15.0, 7.0, 21.0, 97.0, 11.0, 8.0, 723.0, 265.0, 283.0, 445.0, 25.0, 3637.0, 178.0, 24.0, 81.0, 4.0, 5.0, 4608.0, 41.0, 0.0, 5.0, 276.0, 46.0, 63.0, 301.0, 9.0, 14.0, 1458.0, 495.0, 11.0, 25.0], "type": "choropleth"}], "name": "2021-09-12"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-09-19<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Austria", "Belgium", "Britain", "Bulgaria", "Chile", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iran", "Israel", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "South Africa", "Spain", "Sweden", "Switzerland"], "name": "", "z": [56.0, 43.0, 1014.0, 506.0, 127.0, 60.0, 10.0, 17.0, 17.0, 213.0, 14.0, 10.0, 572.0, 352.0, 297.0, 330.0, 37.0, 2871.0, 158.0, 31.0, 105.0, 0.0, 4.0, 3755.0, 56.0, 0.0, 14.0, 302.0, 63.0, 46.0, 524.0, 11.0, 29.0, 1297.0, 493.0, 72.0, 48.0], "type": "choropleth"}], "name": "2021-09-19"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-09-26<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Austria", "Belgium", "Britain", "Bulgaria", "Chile", "Croatia", "Czech Republic", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Guatemala", "Hungary", "Iran", "Israel", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Romania", "Slovenia", "South Africa", "Spain", "Sweden", "Switzerland"], "name": "", "z": [74.0, 57.0, 990.0, 461.0, 81.0, 95.0, 23.0, 14.0, 59.0, 18.0, 12.0, 486.0, 432.0, 213.0, 335.0, 28.0, 2178.0, 134.0, 44.0, 120.0, 1.0, 4.0, 3947.0, 50.0, 0.0, 9.0, 226.0, 84.0, 47.0, 827.0, 39.0, 878.0, 446.0, 46.0, 43.0], "type": "choropleth"}], "name": "2021-09-26"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-10-03<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Austria", "Belgium", "Britain", "Bulgaria", "Chile", "Croatia", "Denmark", "Ecuador", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Iran", "Israel", "Latvia", "Lithuania", "Luxembourg", "Malta", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Slovenia", "South Africa", "Spain", "Sweden", "Switzerland"], "name": "", "z": [57.0, 58.0, 809.0, 549.0, 54.0, 83.0, 21.0, 71.0, 19.0, 9.0, 413.0, 389.0, 241.0, 48.0, 1749.0, 146.0, 57.0, 150.0, 0.0, 2.0, 3142.0, 41.0, 0.0, 11.0, 193.0, 123.0, 39.0, 37.0, 728.0, 234.0, 47.0, 50.0], "type": "choropleth"}], "name": "2021-10-03"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-10-10<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Austria", "Belgium", "Britain", "Bulgaria", "Chile", "Denmark", "Ecuador", "Estonia", "Finland", "Germany", "Iran", "Israel", "Latvia", "Lithuania", "Mexico", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Portugal", "Slovenia", "South Africa", "Spain", "Sweden", "Switzerland"], "name": "", "z": [70.0, 63.0, 801.0, 618.0, 70.0, 6.0, 45.0, 31.0, 18.0, 415.0, 1483.0, 99.0, 60.0, 168.0, 3494.0, 32.0, 1.0, 10.0, 190.0, 174.0, 48.0, 39.0, 537.0, 315.0, 37.0, 27.0], "type": "choropleth"}], "name": "2021-10-10"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-10-17<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Austria", "Belgium", "Britain", "Bulgaria", "Chile", "Denmark", "Ecuador", "Estonia", "Finland", "Germany", "Iran", "Latvia", "Lithuania", "Netherlands", "New Zealand", "Norway", "Peru", "Portugal", "Slovenia", "South Africa", "Spain", "Sweden"], "name": "", "z": [71.0, 72.0, 858.0, 618.0, 45.0, 16.0, 63.0, 32.0, 20.0, 419.0, 1483.0, 89.0, 208.0, 50.0, 0.0, 13.0, 168.0, 56.0, 25.0, 295.0, 196.0, 51.0], "type": "choropleth"}], "name": "2021-10-17"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-10-24<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Austria", "Britain", "Bulgaria", "Chile", "Denmark", "Ecuador", "Estonia", "Finland", "Germany", "Iran", "Latvia", "Netherlands", "New Zealand", "Norway", "Peru", "Portugal", "South Africa", "Spain", "Sweden"], "name": "", "z": [84.0, 953.0, 799.0, 60.0, 14.0, 53.0, 35.0, 23.0, 489.0, 1148.0, 138.0, 72.0, 0.0, 10.0, 209.0, 36.0, 313.0, 158.0, 37.0], "type": "choropleth"}], "name": "2021-10-24"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-10-31<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Denmark", "Ecuador", "Estonia", "Iran", "Netherlands", "Peru", "South Africa"], "name": "", "z": [15.0, 6.0, 70.0, 1080.0, 98.0, 194.0, 252.0], "type": "choropleth"}], "name": "2021-10-31"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-11-07<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Iran", "Peru", "South Africa"], "name": "", "z": [996.0, 184.0, 155.0], "type": "choropleth"}], "name": "2021-11-07"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-01-08<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [0.0], "type": "choropleth"}], "name": "2020-01-08"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-01-15<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [0.0], "type": "choropleth"}], "name": "2020-01-15"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-01-22<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [0.0], "type": "choropleth"}], "name": "2020-01-22"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-01-29<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [0.0], "type": "choropleth"}], "name": "2020-01-29"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-02-05<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [0.0], "type": "choropleth"}], "name": "2020-02-05"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-02-12<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [0.0], "type": "choropleth"}], "name": "2020-02-12"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-02-19<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [0.0], "type": "choropleth"}], "name": "2020-02-19"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-02-26<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [0.0], "type": "choropleth"}], "name": "2020-02-26"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-03-04<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [0.0], "type": "choropleth"}], "name": "2020-03-04"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-03-11<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [0.0], "type": "choropleth"}], "name": "2020-03-11"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-03-18<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [0.5], "type": "choropleth"}], "name": "2020-03-18"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-03-25<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [21.5], "type": "choropleth"}], "name": "2020-03-25"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-04-01<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [85.0], "type": "choropleth"}], "name": "2020-04-01"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-04-08<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [255.5], "type": "choropleth"}], "name": "2020-04-08"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-04-15<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [339.0], "type": "choropleth"}], "name": "2020-04-15"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-04-22<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [428.0], "type": "choropleth"}], "name": "2020-04-22"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-04-29<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [366.5], "type": "choropleth"}], "name": "2020-04-29"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-05-06<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [264.0], "type": "choropleth"}], "name": "2020-05-06"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-05-13<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [187.0], "type": "choropleth"}], "name": "2020-05-13"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-05-20<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [152.5], "type": "choropleth"}], "name": "2020-05-20"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-05-27<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [99.0], "type": "choropleth"}], "name": "2020-05-27"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-06-03<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [94.0], "type": "choropleth"}], "name": "2020-06-03"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-06-10<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [72.0], "type": "choropleth"}], "name": "2020-06-10"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-06-17<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [56.5], "type": "choropleth"}], "name": "2020-06-17"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-06-24<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [79.5], "type": "choropleth"}], "name": "2020-06-24"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-07-01<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [65.0], "type": "choropleth"}], "name": "2020-07-01"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-07-08<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [64.5], "type": "choropleth"}], "name": "2020-07-08"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-07-15<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [71.0], "type": "choropleth"}], "name": "2020-07-15"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-07-22<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [62.0], "type": "choropleth"}], "name": "2020-07-22"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-07-29<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [59.5], "type": "choropleth"}], "name": "2020-07-29"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-08-05<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [60.0], "type": "choropleth"}], "name": "2020-08-05"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-08-12<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [54.0], "type": "choropleth"}], "name": "2020-08-12"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-08-19<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [71.5], "type": "choropleth"}], "name": "2020-08-19"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-08-26<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [73.5], "type": "choropleth"}], "name": "2020-08-26"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-09-02<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [127.0], "type": "choropleth"}], "name": "2020-09-02"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-09-09<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [182.5], "type": "choropleth"}], "name": "2020-09-09"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-09-16<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [202.0], "type": "choropleth"}], "name": "2020-09-16"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-09-23<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [226.5], "type": "choropleth"}], "name": "2020-09-23"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-09-30<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [245.5], "type": "choropleth"}], "name": "2020-09-30"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-10-07<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [211.5], "type": "choropleth"}], "name": "2020-10-07"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-10-14<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [202.0], "type": "choropleth"}], "name": "2020-10-14"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-10-21<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [244.0], "type": "choropleth"}], "name": "2020-10-21"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-10-28<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [252.5], "type": "choropleth"}], "name": "2020-10-28"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-11-04<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [265.5], "type": "choropleth"}], "name": "2020-11-04"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-11-11<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [289.0], "type": "choropleth"}], "name": "2020-11-11"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-11-18<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [322.5], "type": "choropleth"}], "name": "2020-11-18"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-11-25<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [484.0], "type": "choropleth"}], "name": "2020-11-25"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-12-02<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [632.0], "type": "choropleth"}], "name": "2020-12-02"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-12-09<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [689.0], "type": "choropleth"}], "name": "2020-12-09"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-12-16<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [783.5], "type": "choropleth"}], "name": "2020-12-16"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-12-23<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [860.5], "type": "choropleth"}], "name": "2020-12-23"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-12-30<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [893.0], "type": "choropleth"}], "name": "2020-12-30"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-01-01<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["Turkey"], "name": "", "z": [246.5], "type": "choropleth"}], "name": "2021-01-01"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-01-04<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "type": "choropleth"}], "name": "2020-01-04"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-01-11<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "type": "choropleth"}], "name": "2020-01-11"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-01-18<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "type": "choropleth"}], "name": "2020-01-18"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-01-25<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "type": "choropleth"}], "name": "2020-01-25"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-02-01<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "type": "choropleth"}], "name": "2020-02-01"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-02-08<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "type": "choropleth"}], "name": "2020-02-08"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-02-15<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "type": "choropleth"}], "name": "2020-02-15"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-02-22<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "type": "choropleth"}], "name": "2020-02-22"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-02-29<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "type": "choropleth"}], "name": "2020-02-29"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-03-07<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 19.0], "type": "choropleth"}], "name": "2020-03-07"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-03-14<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 24.0, 0.0, 0.0, 0.0, 39.0], "type": "choropleth"}], "name": "2020-03-14"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-03-21<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [0.0, 0.0, 1.0, 0.0, 23.0, 5.0, 5.0, 0.0, 1.0, 7.0, 19.0, 0.0, 0.0, 6.0, 9.0, 0.0, 1.0, 3.0, 19.0, 0.0, 3.0, 2.0, 5.0, 1.0, 1.0, 3.0, 0.0, 0.0, 2.0, 0.0, 15.0, 0.0, 75.0, 0.0, 0.0, 3.0, 1.0, 3.0, 2.0, 0.0, 3.0, 0.0, 1.0, 5.0, 0.0, 2.0, 1.0, 53.0, 0.0, 4.0, 0.0, 284.0], "type": "choropleth"}], "name": "2020-03-21"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-03-28<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [3.0, 2.0, 14.0, 5.0, 94.0, 38.0, 28.0, 5.0, 4.0, 46.0, 58.0, 0.0, 5.0, 41.0, 38.0, 3.0, 3.0, 7.0, 117.0, 1.0, 7.0, 42.0, 99.0, 4.0, 5.0, 7.0, 1.0, 2.0, 12.0, 3.0, 124.0, 2.0, 914.0, 4.0, 1.0, 22.0, 14.0, 9.0, 32.0, 2.0, 12.0, 0.0, 5.0, 25.0, 3.0, 10.0, 15.0, 91.0, 0.0, 9.0, 0.0, 1988.0], "type": "choropleth"}], "name": "2020-03-28"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-04-04<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [23.0, 3.0, 37.0, 8.0, 199.0, 82.0, 132.0, 9.0, 16.0, 139.0, 129.0, 3.0, 5.0, 197.0, 138.0, 8.0, 16.0, 30.0, 271.0, 9.0, 43.0, 172.0, 436.0, 19.0, 29.0, 13.0, 4.0, 4.0, 32.0, 6.0, 697.0, 9.0, 3532.0, 20.0, 2.0, 77.0, 27.0, 13.0, 102.0, 13.0, 25.0, 1.0, 37.0, 77.0, 5.0, 4.0, 32.0, 126.0, 2.0, 43.0, 0.0, 7056.0], "type": "choropleth"}], "name": "2020-04-04"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-04-11<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [34.0, 3.0, 56.0, 12.0, 309.0, 148.0, 329.0, 19.0, 1.0, 251.0, 223.0, 5.0, 17.0, 433.0, 207.0, 23.0, 34.0, 54.0, 395.0, 9.0, 149.0, 470.0, 851.0, 40.0, 57.0, 87.0, 1.0, 8.0, 56.0, 14.0, 1346.0, 9.0, 6588.0, 55.0, 4.0, 145.0, 52.0, 25.0, 355.0, 41.0, 32.0, 3.0, 58.0, 147.0, 10.0, 8.0, 80.0, 181.0, 4.0, 81.0, 0.0, 13519.0], "type": "choropleth"}], "name": "2020-04-11"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-04-18<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [50.0, 1.0, 68.0, 12.0, 515.0, 137.0, 592.0, 34.0, 69.0, 304.0, 243.0, 1.0, 17.0, 582.0, 224.0, 40.0, 32.0, 50.0, 465.0, 13.0, 172.0, 874.0, 918.0, 57.0, 59.0, 67.0, 6.0, 0.0, 66.0, 15.0, 1888.0, 33.0, 5117.0, 84.0, 2.0, 204.0, 37.0, 21.0, 345.0, 81.0, 47.0, 2.0, 44.0, 199.0, 7.0, 12.0, 128.0, 133.0, 12.0, 74.0, 2.0, 14155.0], "type": "choropleth"}], "name": "2020-04-18"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-04-25<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [102.0, 0.0, 96.0, 10.0, 543.0, 259.0, 775.0, 45.0, 74.0, 304.0, 229.0, 0.0, 12.0, 616.0, 227.0, 38.0, 33.0, 61.0, 375.0, 18.0, 321.0, 1170.0, 966.0, 122.0, 69.0, 98.0, 4.0, 0.0, 48.0, 22.0, 1795.0, 40.0, 5242.0, 125.0, 7.0, 258.0, 63.0, 15.0, 700.0, 78.0, 47.0, 3.0, 33.0, 169.0, 16.0, 9.0, 179.0, 114.0, 15.0, 55.0, 5.0, 15605.0], "type": "choropleth"}], "name": "2020-04-25"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-05-02<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [75.0, 0.0, 74.0, 25.0, 495.0, 162.0, 575.0, 56.0, 75.0, 310.0, 263.0, 7.0, 7.0, 679.0, 272.0, 63.0, 22.0, 43.0, 302.0, 6.0, 397.0, 1116.0, 745.0, 150.0, 71.0, 78.0, 2.0, 64.0, 53.0, 24.0, 1876.0, 45.0, 2513.0, 131.0, 8.0, 312.0, 44.0, 22.0, 881.0, 81.0, 101.0, 11.0, 31.0, 224.0, 8.0, 5.0, 181.0, 91.0, 16.0, 68.0, 0.0, 12860.0], "type": "choropleth"}], "name": "2020-05-02"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-05-09<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [101.0, 1.0, 183.0, 16.0, 504.0, 135.0, 496.0, 34.0, 71.0, 349.0, 225.0, 1.0, 4.0, 793.0, 247.0, 76.0, 31.0, 56.0, 241.0, 8.0, 337.0, 994.0, 505.0, 164.0, 128.0, 121.0, 0.0, 18.0, 49.0, 47.0, 1373.0, 53.0, 2737.0, 123.0, 10.0, 310.0, 32.0, 18.0, 1252.0, 122.0, 62.0, 13.0, 33.0, 202.0, 19.0, 3.0, 211.0, 91.0, 3.0, 64.0, 0.0, 12666.0], "type": "choropleth"}], "name": "2020-05-09"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-05-16<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [94.0, 0.0, 147.0, 10.0, 519.0, 225.0, 406.0, 84.0, 64.0, 250.0, 194.0, 0.0, 6.0, 780.0, 234.0, 95.0, 21.0, 30.0, 225.0, 6.0, 343.0, 865.0, 354.0, 141.0, 91.0, 116.0, 0.0, 26.0, 44.0, 40.0, 1134.0, 62.0, 1246.0, 105.0, 8.0, 279.0, 18.0, 10.0, 534.0, 71.0, 51.0, 10.0, 53.0, 256.0, 10.0, 0.0, 174.0, 79.0, 12.0, 54.0, 0.0, 9576.0], "type": "choropleth"}], "name": "2020-05-16"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-05-23<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [69.0, 0.0, 119.0, 17.0, 527.0, 133.0, 336.0, 38.0, 52.0, 269.0, 234.0, 0.0, 5.0, 663.0, 216.0, 98.0, 10.0, 57.0, 149.0, 7.0, 315.0, 599.0, 340.0, 150.0, 106.0, 86.0, 0.0, 23.0, 44.0, 37.0, 831.0, 55.0, 920.0, 84.0, 10.0, 341.0, 23.0, 10.0, 893.0, 108.0, 45.0, 6.0, 30.0, 199.0, 19.0, 1.0, 156.0, 51.0, 7.0, 55.0, 5.0, 8548.0], "type": "choropleth"}], "name": "2020-05-23"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-05-30<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [67.0, 0.0, 101.0, 18.0, 406.0, 117.0, 238.0, 37.0, 35.0, 214.0, 180.0, 0.0, 4.0, 540.0, 131.0, 83.0, 9.0, 40.0, 118.0, 12.0, 320.0, 464.0, 244.0, 176.0, 107.0, 94.0, 1.0, 21.0, 22.0, 34.0, 553.0, 43.0, 671.0, 144.0, 7.0, 195.0, 22.0, 6.0, 440.0, 121.0, 58.0, 10.0, 39.0, 142.0, 11.0, 1.0, 206.0, 68.0, 2.0, 81.0, 3.0, 6656.0], "type": "choropleth"}], "name": "2020-05-30"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-06-06<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [66.0, 0.0, 139.0, 20.0, 463.0, 82.0, 143.0, 29.0, 21.0, 242.0, 173.0, 0.0, 1.0, 533.0, 167.0, 71.0, 19.0, 38.0, 134.0, 9.0, 361.0, 521.0, 426.0, 143.0, 88.0, 37.0, 1.0, 21.0, 25.0, 41.0, 472.0, 41.0, 550.0, 115.0, 13.0, 224.0, 14.0, 8.0, 394.0, 54.0, 62.0, 5.0, 49.0, 169.0, 13.0, 0.0, 89.0, 35.0, 11.0, 57.0, 2.0, 6391.0], "type": "choropleth"}], "name": "2020-06-06"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-06-13<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [84.0, 2.0, 148.0, 24.0, 451.0, 73.0, 131.0, 29.0, 28.0, 237.0, 269.0, 0.0, 3.0, 426.0, 121.0, 51.0, 11.0, 29.0, 78.0, 2.0, 184.0, 287.0, 122.0, 114.0, 78.0, 70.0, 0.0, 29.0, 9.0, 35.0, 482.0, 39.0, 428.0, 114.0, 2.0, 184.0, 12.0, 13.0, 280.0, 61.0, 53.0, 10.0, 52.0, 141.0, 18.0, 0.0, 83.0, 59.0, 4.0, 46.0, 1.0, 5207.0], "type": "choropleth"}], "name": "2020-06-13"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-06-20<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [70.0, 0.0, 154.0, 47.0, 431.0, 49.0, 65.0, 15.0, 20.0, 219.0, 192.0, 0.0, 3.0, 336.0, 115.0, 32.0, 19.0, 26.0, 99.0, 2.0, 141.0, 251.0, 75.0, 91.0, 49.0, 78.0, 2.0, 28.0, 23.0, 21.0, 269.0, 35.0, 303.0, 108.0, 2.0, 143.0, 9.0, 15.0, 209.0, 61.0, 45.0, 6.0, 51.0, 207.0, 16.0, 1.0, 68.0, 51.0, 0.0, 52.0, 2.0, 4306.0], "type": "choropleth"}], "name": "2020-06-20"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-06-27<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [81.0, 0.0, 242.0, 35.0, 412.0, 28.0, 60.0, 73.0, 17.0, 246.0, 145.0, 1.0, 2.0, 248.0, 78.0, 23.0, 6.0, 30.0, 88.0, 2.0, 108.0, 214.0, 67.0, 45.0, 97.0, 41.0, 2.0, 23.0, 15.0, 28.0, 2092.0, 25.0, 266.0, 106.0, 2.0, 107.0, 16.0, 14.0, 183.0, 33.0, 64.0, 10.0, 61.0, 204.0, 12.0, 0.0, 120.0, 47.0, 5.0, 34.0, 0.0, 5858.0], "type": "choropleth"}], "name": "2020-06-27"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-07-04<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [88.0, 2.0, 226.0, 27.0, 432.0, 27.0, 24.0, 5.0, 9.0, 312.0, 83.0, 1.0, 1.0, 141.0, 70.0, 17.0, 7.0, 31.0, 92.0, 5.0, 79.0, 130.0, 65.0, 49.0, 72.0, 26.0, 1.0, 17.0, 30.0, 9.0, 241.0, 22.0, 751.0, 77.0, 2.0, 103.0, 14.0, 10.0, 147.0, 33.0, 99.0, 6.0, 52.0, 242.0, 14.0, 0.0, 124.0, 44.0, 1.0, 19.0, 0.0, 4079.0], "type": "choropleth"}], "name": "2020-07-04"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-07-11<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [107.0, 0.0, 346.0, 33.0, 696.0, 24.0, 13.0, 3.0, 11.0, 495.0, 137.0, 0.0, 10.0, 154.0, 69.0, 27.0, 13.0, 37.0, 126.0, 5.0, 79.0, 139.0, 95.0, 33.0, 123.0, 47.0, 6.0, 1.0, 59.0, 15.0, 336.0, 30.0, 153.0, 98.0, 7.0, 129.0, 23.0, 19.0, 148.0, 16.0, 134.0, 12.0, 99.0, 504.0, 31.0, 0.0, 118.0, 70.0, 2.0, 25.0, 1.0, 4858.0], "type": "choropleth"}], "name": "2020-07-11"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-07-18<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [172.0, 4.0, 579.0, 38.0, 676.0, 27.0, 48.0, 8.0, 10.0, 698.0, 171.0, 5.0, 17.0, 122.0, 64.0, 39.0, 17.0, 45.0, 104.0, 3.0, 69.0, 109.0, 51.0, 39.0, 116.0, 61.0, 8.0, 15.0, 57.0, 5.0, 174.0, 26.0, 157.0, 136.0, 3.0, 96.0, 30.0, 25.0, 110.0, 14.0, 177.0, 7.0, 100.0, 753.0, 31.0, 0.0, 61.0, 20.0, 4.0, 22.0, 3.0, 5326.0], "type": "choropleth"}], "name": "2020-07-18"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-07-25<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [170.0, 2.0, 556.0, 42.0, 726.0, 42.0, 17.0, 56.0, 3.0, 882.0, 327.0, 2.0, 27.0, 107.0, 71.0, 39.0, 19.0, 29.0, 204.0, 2.0, 63.0, 91.0, 36.0, 7.0, 134.0, 52.0, 9.0, 17.0, 86.0, 13.0, 77.0, 38.0, 129.0, 149.0, 9.0, 165.0, 45.0, 29.0, 107.0, 4.0, 295.0, 6.0, 136.0, 1020.0, 31.0, 0.0, 48.0, 50.0, 3.0, 48.0, 1.0, 6221.0], "type": "choropleth"}], "name": "2020-07-25"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-08-01<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [147.0, 4.0, 461.0, 59.0, 933.0, 50.0, 19.0, 6.0, 4.0, 1245.0, 330.0, 0.0, 50.0, 106.0, 73.0, 46.0, 34.0, 44.0, 232.0, 4.0, 66.0, 116.0, 57.0, 61.0, 212.0, 71.0, 15.0, 16.0, 100.0, 7.0, 54.0, 44.0, 95.0, 186.0, 4.0, 218.0, 53.0, 39.0, 90.0, -5.0, 271.0, 12.0, 103.0, 1950.0, 36.0, 1.0, 140.0, 98.0, 13.0, 56.0, 0.0, 8026.0], "type": "choropleth"}], "name": "2020-08-01"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-08-08<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [165.0, 2.0, 390.0, 77.0, 924.0, 13.0, 9.0, 5.0, 5.0, 1087.0, 361.0, 5.0, 39.0, 128.0, 63.0, 53.0, 22.0, 32.0, 254.0, 2.0, 69.0, 95.0, 63.0, 42.0, 165.0, 48.0, 14.0, 13.0, 117.0, 3.0, 39.0, 30.0, 60.0, 196.0, 9.0, 153.0, 54.0, 30.0, 109.0, 5.0, 233.0, 12.0, 148.0, 1506.0, 25.0, 1.0, 107.0, 80.0, 15.0, 49.0, 2.0, 7128.0], "type": "choropleth"}], "name": "2020-08-08"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-08-15<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [128.0, 2.0, 355.0, 65.0, 937.0, 39.0, 12.0, 3.0, 7.0, 1236.0, 483.0, 9.0, 34.0, 95.0, 87.0, 48.0, 27.0, 38.0, 218.0, 2.0, 65.0, 99.0, 66.0, 51.0, 223.0, 45.0, 7.0, 16.0, 120.0, 4.0, 41.0, 30.0, 67.0, 183.0, 9.0, 156.0, 54.0, 31.0, 152.0, 7.0, 344.0, 6.0, 130.0, 1498.0, 28.0, 0.0, 59.0, 94.0, 29.0, 42.0, 1.0, 7482.0], "type": "choropleth"}], "name": "2020-08-15"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-08-22<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [115.0, 3.0, 264.0, 74.0, 894.0, 22.0, 7.0, 7.0, 7.0, 929.0, 423.0, 7.0, 37.0, 148.0, 80.0, 57.0, 23.0, 62.0, 239.0, 3.0, 48.0, 101.0, 69.0, 62.0, 155.0, 79.0, 8.0, 15.0, 128.0, 6.0, 33.0, 32.0, 53.0, 178.0, 14.0, 151.0, 68.0, 31.0, 111.0, 7.0, 233.0, 7.0, 218.0, 1423.0, 23.0, 0.0, 62.0, 91.0, 16.0, 43.0, 9.0, 6875.0], "type": "choropleth"}], "name": "2020-08-22"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-08-29<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [141.0, 6.0, 251.0, 98.0, 789.0, 26.0, 5.0, 4.0, 1.0, 831.0, 484.0, 15.0, 52.0, 134.0, 284.0, 78.0, 18.0, 49.0, 195.0, 2.0, 62.0, 115.0, 57.0, 53.0, 190.0, 71.0, 14.0, 15.0, 105.0, 3.0, -10.0, 26.0, 58.0, 162.0, 6.0, 286.0, 72.0, 38.0, 95.0, 12.0, 205.0, 8.0, 162.0, 1154.0, 21.0, 0.0, 125.0, 48.0, 36.0, 38.0, 0.0, 6690.0], "type": "choropleth"}], "name": "2020-08-29"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-09-05<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [123.0, 5.0, 200.0, 110.0, 802.0, 22.0, 3.0, 4.0, 6.0, 706.0, 401.0, 22.0, 26.0, 158.0, 77.0, 52.0, 33.0, 72.0, 131.0, 2.0, 53.0, 80.0, 94.0, 37.0, 144.0, 143.0, 12.0, 12.0, 86.0, 0.0, 52.0, 31.0, 57.0, 206.0, 14.0, -5.0, 53.0, 25.0, 89.0, 11.0, 179.0, 6.0, 137.0, 987.0, 13.0, 0.0, 109.0, 48.0, 31.0, 49.0, 5.0, 5713.0], "type": "choropleth"}], "name": "2020-09-05"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-09-12<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [75.0, 2.0, 108.0, 71.0, 626.0, 22.0, 12.0, 5.0, 5.0, 789.0, 310.0, 13.0, 31.0, 129.0, 75.0, 48.0, 41.0, 64.0, 160.0, 0.0, 38.0, 80.0, 105.0, 55.0, 112.0, 65.0, 17.0, 31.0, 61.0, 3.0, 42.0, 21.0, 46.0, 158.0, 12.0, 155.0, 49.0, 25.0, 77.0, 11.0, 163.0, 10.0, 202.0, 735.0, 13.0, 0.0, 45.0, 38.0, 22.0, 41.0, 0.0, 5018.0], "type": "choropleth"}], "name": "2020-09-12"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-09-19<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [87.0, 1.0, 152.0, 228.0, 654.0, 25.0, 12.0, 8.0, 0.0, 687.0, 312.0, 23.0, 26.0, 141.0, 66.0, 56.0, 76.0, 51.0, 140.0, 5.0, 40.0, 99.0, 58.0, 57.0, 128.0, 89.0, 23.0, 8.0, 79.0, 3.0, 37.0, 26.0, 54.0, 188.0, 25.0, 201.0, 44.0, 19.0, 119.0, 13.0, 148.0, 17.0, 152.0, 705.0, 7.0, 0.0, 268.0, 45.0, 43.0, 32.0, 7.0, 5484.0], "type": "choropleth"}], "name": "2020-09-19"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-09-26<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [64.0, 7.0, 155.0, 104.0, 547.0, 27.0, 9.0, 12.0, 8.0, 735.0, 315.0, 11.0, 19.0, 152.0, 74.0, 47.0, 34.0, 46.0, 90.0, 1.0, 49.0, 96.0, 75.0, 41.0, 102.0, 270.0, 15.0, 27.0, 54.0, 1.0, 39.0, 23.0, 34.0, 205.0, 19.0, 128.0, 61.0, 19.0, 147.0, 23.0, 135.0, 18.0, 158.0, 634.0, 8.0, 0.0, 154.0, 62.0, 24.0, 40.0, 1.0, 5119.0], "type": "choropleth"}], "name": "2020-09-26"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-10-03<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [57.0, 6.0, 83.0, 122.0, 608.0, 17.0, 12.0, 12.0, 5.0, 606.0, 220.0, 22.0, 20.0, 186.0, 92.0, 66.0, 66.0, 51.0, 125.0, 2.0, 31.0, 116.0, 80.0, 69.0, 104.0, 107.0, 15.0, 28.0, 38.0, 3.0, 32.0, 20.0, 80.0, 189.0, 60.0, 185.0, 47.0, 20.0, 96.0, 14.0, 119.0, 30.0, 186.0, 507.0, 28.0, 0.0, 126.0, 45.0, 25.0, 91.0, 3.0, 4872.0], "type": "choropleth"}], "name": "2020-10-03"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-10-10<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [106.0, 2.0, 54.0, 145.0, 447.0, 55.0, 17.0, 8.0, 7.0, 736.0, 259.0, 15.0, 27.0, 201.0, 113.0, 77.0, 65.0, 44.0, 55.0, 1.0, 34.0, 80.0, 95.0, 58.0, 81.0, 253.0, 23.0, 21.0, 39.0, 13.0, 36.0, 17.0, 83.0, 136.0, 65.0, 72.0, 44.0, 34.0, 73.0, 12.0, 109.0, 38.0, 198.0, 534.0, 37.0, 0.0, 84.0, 47.0, 24.0, 86.0, 0.0, 4860.0], "type": "choropleth"}], "name": "2020-10-10"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-10-17<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [124.0, 7.0, 65.0, 132.0, 375.0, 64.0, 12.0, 12.0, 5.0, 553.0, 214.0, 18.0, 22.0, 217.0, 136.0, 72.0, 96.0, 63.0, 85.0, 3.0, 49.0, 136.0, 121.0, 20.0, 75.0, 162.0, 31.0, 29.0, 48.0, 11.0, 33.0, 22.0, 85.0, 164.0, 63.0, 70.0, 73.0, 23.0, 194.0, 12.0, 86.0, 29.0, 145.0, 457.0, 28.0, 0.0, 68.0, 49.0, 18.0, 116.0, 4.0, 4696.0], "type": "choropleth"}], "name": "2020-10-17"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-10-24<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [78.0, 1.0, 45.0, 113.0, 381.0, 42.0, 35.0, 16.0, 1.0, 500.0, 201.0, 26.0, 42.0, 289.0, 200.0, 103.0, 116.0, 92.0, 87.0, 0.0, 45.0, 116.0, 182.0, 95.0, 84.0, 253.0, 54.0, 48.0, 36.0, 7.0, 77.0, 36.0, 81.0, 63.0, 49.0, 139.0, 77.0, 33.0, 188.0, 23.0, 156.0, 51.0, 197.0, 472.0, 27.0, 0.0, 156.0, 47.0, 23.0, 196.0, 11.0, 5390.0], "type": "choropleth"}], "name": "2020-10-24"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-10-31<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [98.0, 14.0, 109.0, 127.0, 314.0, 58.0, 38.0, 28.0, 4.0, 344.0, 163.0, 7.0, 57.0, 228.0, 213.0, 86.0, 54.0, 79.0, 91.0, 1.0, 58.0, 152.0, 174.0, 229.0, 93.0, 221.0, 81.0, 49.0, 34.0, 10.0, 67.0, 53.0, 116.0, 175.0, 76.0, 95.0, 92.0, 35.0, 158.0, 27.0, 139.0, 59.0, 251.0, 566.0, 36.0, 0.0, 60.0, 79.0, 31.0, 259.0, 19.0, 5607.0], "type": "choropleth"}], "name": "2020-10-31"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-11-07<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [115.0, 1.0, 169.0, 142.0, 305.0, 111.0, 55.0, 7.0, 8.0, 339.0, 219.0, 1.0, 54.0, 444.0, 260.0, 100.0, 137.0, 76.0, 82.0, 5.0, 56.0, 134.0, 181.0, 149.0, 94.0, 94.0, 70.0, 56.0, 71.0, 6.0, 77.0, 86.0, 134.0, 381.0, 104.0, 205.0, 92.0, 41.0, 203.0, 23.0, 83.0, 98.0, 236.0, 674.0, 54.0, 1.0, 59.0, 72.0, 48.0, 272.0, 17.0, 6501.0], "type": "choropleth"}], "name": "2020-11-07"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-11-14<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [165.0, 13.0, 152.0, 89.0, 288.0, 137.0, 67.0, 20.0, 4.0, 388.0, 257.0, 2.0, 68.0, 515.0, 296.0, 154.0, 88.0, 98.0, 98.0, 11.0, 80.0, 168.0, 494.0, 281.0, 98.0, 219.0, 66.0, 77.0, 57.0, 10.0, 123.0, 104.0, 192.0, 171.0, 98.0, 208.0, 87.0, 30.0, 259.0, 24.0, 95.0, 98.0, 290.0, 767.0, 53.0, 0.0, 90.0, 77.0, 72.0, 323.0, 19.0, 7640.0], "type": "choropleth"}], "name": "2020-11-14"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-11-21<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [211.0, 0.0, 157.0, 174.0, 345.0, 263.0, 91.0, 12.0, 12.0, 442.0, 166.0, 9.0, 95.0, 762.0, 359.0, 186.0, 156.0, 125.0, 348.0, 13.0, 130.0, 195.0, 503.0, 336.0, 117.0, 180.0, 87.0, 122.0, 102.0, 9.0, 198.0, 142.0, 230.0, 286.0, 108.0, 28.0, 87.0, 60.0, 415.0, 31.0, 90.0, 120.0, 324.0, 827.0, 64.0, 3.0, 120.0, 100.0, 65.0, 327.0, 41.0, 9373.0], "type": "choropleth"}], "name": "2020-11-21"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-11-28<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [115.0, 0.0, 168.0, 107.0, 536.0, 196.0, 133.0, 21.0, 8.0, 507.0, 158.0, 9.0, 68.0, 707.0, 417.0, 192.0, 119.0, 102.0, 174.0, 15.0, 161.0, 188.0, 588.0, 280.0, 122.0, 269.0, 71.0, 81.0, 105.0, 6.0, 219.0, 177.0, 338.0, 133.0, 81.0, 657.0, 114.0, 77.0, 586.0, 33.0, 153.0, 201.0, 333.0, 1013.0, 88.0, 5.0, 144.0, 81.0, 80.0, 333.0, 51.0, 10520.0], "type": "choropleth"}], "name": "2020-11-28"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-12-05<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [304.0, 0.0, 301.0, 184.0, 746.0, 373.0, 185.0, 24.0, 17.0, 632.0, 196.0, 21.0, 119.0, 1041.0, 544.0, 279.0, 257.0, 178.0, 139.0, 35.0, 221.0, 277.0, 855.0, 398.0, 170.0, 360.0, 64.0, 211.0, 170.0, 45.0, 339.0, 211.0, 461.0, 377.0, 92.0, 544.0, 157.0, 131.0, 987.0, 62.0, 164.0, 148.0, 365.0, 1194.0, 76.0, 12.0, 143.0, 137.0, 111.0, 415.0, 42.0, 14514.0], "type": "choropleth"}], "name": "2020-12-05"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-12-12<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [205.0, 72.0, 397.0, 290.0, 1092.0, 515.0, 217.0, 22.0, 18.0, 716.0, 236.0, 8.0, 137.0, 998.0, 550.0, 582.0, 286.0, 130.0, 220.0, 31.0, 276.0, 354.0, 878.0, 440.0, 231.0, 329.0, 83.0, 171.0, 214.0, 41.0, 427.0, 175.0, 591.0, 278.0, 57.0, 530.0, 168.0, 110.0, 1174.0, 74.0, 168.0, 153.0, 499.0, 1297.0, 101.0, 16.0, 215.0, 47.0, 137.0, 340.0, 64.0, 16360.0], "type": "choropleth"}], "name": "2020-12-12"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-12-19<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [310.0, 7.0, 615.0, 281.0, 1612.0, 468.0, 218.0, 43.0, 17.0, 688.0, 231.0, 0.0, 106.0, 944.0, 581.0, 242.0, 269.0, 179.0, 228.0, 35.0, 305.0, 350.0, 878.0, 421.0, 210.0, 427.0, 48.0, 110.0, 222.0, 50.0, 442.0, 242.0, 827.0, 389.0, 140.0, 550.0, 147.0, 165.0, 1389.0, 98.0, 210.0, 107.0, 562.0, 861.0, 109.0, 13.0, 233.0, 225.0, 156.0, 360.0, 30.0, 17350.0], "type": "choropleth"}], "name": "2020-12-19"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2020-12-26<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [294.0, 16.0, 34.0, 250.0, 1400.0, 253.0, 210.0, 36.0, 32.0, 661.0, 274.0, 0.0, 38.0, 745.0, 463.0, 293.0, 0.0, 163.0, 278.0, 27.0, 280.0, 60.0, 616.0, 327.0, 175.0, 357.0, 55.0, 81.0, 185.0, 6.0, 438.0, 161.0, 657.0, 42.0, 60.0, 453.0, 168.0, 113.0, 1054.0, 116.0, 229.0, 31.0, 481.0, 1775.0, 7.0, 13.0, 193.0, 27.0, 6.0, 283.0, 0.0, 13916.0], "type": "choropleth"}], "name": "2020-12-26"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-01-02<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [188.0, 15.0, 1090.0, 288.0, 2374.0, 320.0, 308.0, 36.0, 30.0, 753.0, 183.0, 0.0, 135.0, 802.0, 609.0, 201.0, 537.0, 164.0, 214.0, 116.0, 275.0, 785.0, 615.0, 270.0, 275.0, 249.0, 52.0, 111.0, 199.0, 113.0, 576.0, 218.0, 1315.0, 667.0, 46.0, 541.0, 170.0, 62.0, 1360.0, 85.0, 260.0, 120.0, 529.0, 1431.0, 140.0, 18.0, 280.0, 56.0, 245.0, 187.0, 87.0, 19700.0], "type": "choropleth"}], "name": "2021-01-02"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-01-09<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [428.0, 8.0, 975.0, 281.0, 2876.0, 278.0, 225.0, 39.0, 21.0, 917.0, 389.0, 38.0, 80.0, 827.0, 556.0, 183.0, 270.0, 178.0, 347.0, 42.0, 287.0, 572.0, 841.0, 286.0, -262.0, 401.0, 82.0, 71.0, 278.0, 77.0, 667.0, 198.0, 1190.0, 534.0, 42.0, 582.0, 211.0, 98.0, 1428.0, 132.0, 374.0, 69.0, 734.0, 1825.0, 97.0, 17.0, 266.0, 447.0, 197.0, 286.0, 51.0, 21036.0], "type": "choropleth"}], "name": "2021-01-09"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-01-16<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [819.0, 5.0, 1212.0, 283.0, 3727.0, 167.0, 269.0, 47.0, 34.0, 1200.0, 749.0, 13.0, 77.0, 684.0, 321.0, 194.0, 354.0, 216.0, 247.0, 62.0, 271.0, 509.0, 509.0, 224.0, 903.0, 306.0, 35.0, 87.0, 300.0, 81.0, 560.0, 177.0, 1341.0, 590.0, 29.0, 535.0, 214.0, 183.0, 1521.0, 71.0, 350.0, 63.0, 651.0, 2140.0, 94.0, 7.0, 323.0, 209.0, 191.0, 294.0, 33.0, 23481.0], "type": "choropleth"}], "name": "2021-01-16"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-01-23<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [539.0, 29.0, 922.0, 270.0, 3400.0, 123.0, 226.0, 20.0, 23.0, 1160.0, 769.0, 8.0, 63.0, 531.0, 405.0, 165.0, 96.0, 294.0, 403.0, 52.0, 309.0, 481.0, 524.0, 176.0, 271.0, 291.0, 53.0, 50.0, 243.0, 44.0, 520.0, 206.0, 1322.0, 570.0, 30.0, 546.0, 279.0, 107.0, 1338.0, 88.0, 371.0, 63.0, 463.0, 2282.0, 97.0, 7.0, 371.0, 269.0, 111.0, 235.0, 49.0, 21264.0], "type": "choropleth"}], "name": "2021-01-23"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-01-30<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [909.0, 3.0, 927.0, 275.0, 3856.0, 136.0, 227.0, 45.0, 39.0, 1196.0, 770.0, 73.0, 57.0, 494.0, 275.0, 165.0, 181.0, 328.0, 318.0, 36.0, 286.0, 467.0, 347.0, 124.0, 267.0, 207.0, 91.0, 40.0, 244.0, 71.0, 521.0, 150.0, 1311.0, 702.0, 11.0, 441.0, 273.0, 73.0, 1076.0, 73.0, 535.0, 79.0, 756.0, 2204.0, 81.0, 3.0, 374.0, 173.0, 143.0, 208.0, 25.0, 21666.0], "type": "choropleth"}], "name": "2021-01-30"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-02-06<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [956.0, 16.0, 910.0, 223.0, 3431.0, 106.0, 168.0, 121.0, 38.0, 1243.0, 756.0, 15.0, 42.0, 382.0, 1785.0, 457.0, 322.0, 306.0, 275.0, 8.0, 147.0, 390.0, 331.0, 103.0, 247.0, 394.0, 78.0, 46.0, 198.0, 56.0, 509.0, 121.0, 1217.0, 639.0, 6.0, 532.0, 257.0, 64.0, 794.0, 74.0, 597.0, 29.0, 889.0, 2156.0, 71.0, 9.0, 325.0, 164.0, 104.0, 162.0, 28.0, 22297.0], "type": "choropleth"}], "name": "2021-02-06"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-02-13<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [722.0, 2.0, 939.0, 191.0, 2788.0, 97.0, 167.0, 80.0, 32.0, 1085.0, 638.0, 9.0, 37.0, 344.0, 397.0, 128.0, 263.0, 252.0, 200.0, 13.0, 202.0, 503.0, 265.0, 80.0, 196.0, 311.0, 17.0, 43.0, 227.0, 32.0, 476.0, 130.0, 1090.0, 527.0, 4.0, 4687.0, 233.0, 92.0, 564.0, 67.0, 302.0, 34.0, 468.0, 1902.0, 57.0, 7.0, 228.0, 226.0, 80.0, 110.0, 23.0, 21567.0], "type": "choropleth"}], "name": "2021-02-13"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-02-20<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [347.0, 8.0, 533.0, 96.0, 2390.0, 75.0, 142.0, 84.0, 15.0, 1126.0, 668.0, 5.0, 22.0, 308.0, 189.0, 100.0, 266.0, 154.0, 165.0, 11.0, 159.0, 355.0, 222.0, 54.0, 92.0, 268.0, 12.0, 44.0, 172.0, 23.0, 395.0, 94.0, 942.0, 444.0, 6.0, 422.0, 161.0, 55.0, 611.0, 94.0, 344.0, 21.0, 199.0, 834.0, 50.0, 7.0, 209.0, 148.0, 17.0, 121.0, 15.0, 13294.0], "type": "choropleth"}], "name": "2021-02-20"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-02-27<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [343.0, 1.0, 501.0, 72.0, 2996.0, 62.0, 104.0, 56.0, 11.0, 926.0, 438.0, 9.0, 34.0, 260.0, 222.0, 135.0, 122.0, 199.0, 148.0, 44.0, 164.0, 288.0, 166.0, 53.0, 116.0, 199.0, 16.0, 31.0, 90.0, 17.0, 404.0, 90.0, 832.0, 316.0, 8.0, 484.0, 224.0, 57.0, 370.0, 137.0, 241.0, 29.0, 274.0, 1526.0, 65.0, 9.0, 1173.0, 134.0, 75.0, 130.0, 9.0, 14410.0], "type": "choropleth"}], "name": "2021-02-27"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-03-06<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [218.0, 12.0, 289.0, 19.0, 1901.0, 40.0, 77.0, 42.0, 25.0, 886.0, 536.0, 5.0, 17.0, 258.0, 168.0, 79.0, 94.0, 181.0, 127.0, 4.0, 86.0, 307.0, 151.0, 69.0, 136.0, 248.0, 24.0, 31.0, 77.0, 11.0, 318.0, 97.0, 646.0, 290.0, 4.0, 637.0, 156.0, 87.0, 381.0, 55.0, 221.0, 8.0, 146.0, 1627.0, 62.0, 3.0, 1140.0, 89.0, 27.0, 71.0, 13.0, 12196.0], "type": "choropleth"}], "name": "2021-03-06"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-03-13<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [181.0, 1.0, 284.0, 87.0, 1513.0, 53.0, 61.0, 32.0, 10.0, 628.0, 285.0, 5.0, 33.0, 174.0, 119.0, 84.0, 52.0, 160.0, 145.0, 17.0, 102.0, 206.0, 115.0, 195.0, 91.0, 149.0, 12.0, 19.0, 60.0, 14.0, 331.0, 54.0, 679.0, 164.0, 9.0, 565.0, 167.0, 26.0, 253.0, 33.0, 152.0, 12.0, 96.0, 1104.0, 49.0, 6.0, 477.0, 85.0, 212.0, 65.0, 9.0, 9435.0], "type": "choropleth"}], "name": "2021-03-13"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-03-20<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [108.0, 4.0, 139.0, 67.0, 883.0, 27.0, 68.0, 26.0, 9.0, 477.0, 275.0, 4.0, 31.0, 133.0, 82.0, 40.0, 15.0, 754.0, 127.0, 4.0, 122.0, 252.0, 134.0, 37.0, 60.0, 63.0, 22.0, 9.0, 71.0, 22.0, 246.0, 38.0, 469.0, 159.0, 3.0, 338.0, 86.0, 38.0, 207.0, 26.0, 108.0, 11.0, 73.0, 881.0, 41.0, 5.0, 209.0, 47.0, 75.0, 55.0, 1.0, 7181.0], "type": "choropleth"}], "name": "2021-03-20"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-03-27<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [89.0, 3.0, 226.0, 49.0, 1628.0, 27.0, 36.0, 22.0, 7.0, 444.0, 336.0, 9.0, 15.0, 171.0, 85.0, 43.0, -21.0, 290.0, 100.0, 8.0, 81.0, 219.0, 141.0, 47.0, 45.0, 74.0, 22.0, 38.0, 61.0, 14.0, 248.0, 36.0, 675.0, 207.0, 6.0, 127.0, 62.0, 16.0, 207.0, 32.0, 129.0, 14.0, 102.0, 740.0, 60.0, 5.0, 62.0, 43.0, 34.0, 29.0, 2.0, 7145.0], "type": "choropleth"}], "name": "2021-03-27"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-04-03<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [110.0, 0.0, 87.0, -1.0, 683.0, 30.0, 37.0, 22.0, 14.0, 497.0, 2710.0, 5.0, 10.0, 2402.0, 64.0, 38.0, 41.0, 139.0, 75.0, 9.0, 81.0, 230.0, 210.0, 50.0, 55.0, 61.0, 29.0, 12.0, 38.0, 16.0, 255.0, 24.0, 468.0, 106.0, 0.0, -270.0, 103.0, 12.0, 201.0, 36.0, 92.0, 14.0, 94.0, 629.0, 15.0, 5.0, 15.0, 58.0, 59.0, 5.0, 4.0, 9679.0], "type": "choropleth"}], "name": "2021-04-03"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-04-10<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [72.0, 11.0, 81.0, 20.0, 684.0, 23.0, 38.0, 7.0, 11.0, 342.0, 278.0, 4.0, 23.0, 142.0, 60.0, 94.0, 5.0, 92.0, 53.0, 5.0, 103.0, 98.0, 303.0, 69.0, 38.0, 0.0, 57.0, 27.0, 171.0, 9.0, 222.0, 27.0, 479.0, 109.0, 3.0, 184.0, 1716.0, 55.0, 172.0, 8.0, 73.0, -1.0, 86.0, 460.0, 27.0, 2.0, 171.0, 43.0, 39.0, 36.0, 1.0, 6832.0], "type": "choropleth"}], "name": "2021-04-10"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-04-17<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [78.0, 20.0, 65.0, 32.0, 577.0, 33.0, 51.0, 29.0, 14.0, 390.0, 269.0, 3.0, 28.0, 168.0, 78.0, 32.0, 18.0, 89.0, 66.0, 14.0, 101.0, 76.0, 371.0, 61.0, 58.0, 120.0, 22.0, -7.0, 33.0, 10.0, 275.0, 25.0, 535.0, 139.0, 8.0, 164.0, 28.0, 17.0, 299.0, 9.0, 48.0, 7.0, 65.0, 388.0, 4.0, 11.0, 106.0, 58.0, 43.0, 35.0, 2.0, 5165.0], "type": "choropleth"}], "name": "2021-04-17"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-04-24<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [61.0, 0.0, 117.0, 24.0, 445.0, 61.0, 52.0, 13.0, 4.0, 408.0, 216.0, 4.0, 14.0, 169.0, 55.0, 26.0, 12.0, 106.0, 54.0, 8.0, 109.0, 89.0, 475.0, 67.0, 22.0, 60.0, 18.0, 29.0, 57.0, 16.0, 233.0, 23.0, 346.0, 136.0, 7.0, 131.0, 19.0, 19.0, 277.0, 13.0, 93.0, 0.0, 80.0, 372.0, 14.0, 2.0, 111.0, 54.0, 37.0, 46.0, 2.0, 4806.0], "type": "choropleth"}], "name": "2021-04-24"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-05-01<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [62.0, 12.0, 70.0, 24.0, 490.0, 53.0, 50.0, 11.0, 7.0, 427.0, 239.0, 5.0, 15.0, 216.0, 67.0, 43.0, 14.0, 77.0, 46.0, 17.0, 97.0, 73.0, 484.0, 82.0, 24.0, 50.0, 11.0, 3.0, 42.0, 22.0, 221.0, 43.0, 314.0, 128.0, 8.0, 162.0, 72.0, 19.0, 315.0, 11.0, 92.0, 14.0, 58.0, 341.0, 23.0, 3.0, 102.0, 65.0, -138.0, 84.0, 2.0, 4772.0], "type": "choropleth"}], "name": "2021-05-01"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-05-08<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [65.0, 2.0, 69.0, 19.0, 542.0, 53.0, 40.0, 10.0, 4.0, 461.0, 156.0, 4.0, 15.0, 216.0, 67.0, 33.0, 34.0, 65.0, 51.0, 6.0, 110.0, 59.0, 484.0, 70.0, 29.0, 78.0, 18.0, 7.0, 34.0, 10.0, 203.0, 31.0, 131.0, 129.0, 5.0, 144.0, 44.0, 27.0, 279.0, 19.0, 80.0, 12.0, 53.0, 293.0, 20.0, 2.0, 108.0, 65.0, 47.0, 63.0, 3.0, 4569.0], "type": "choropleth"}], "name": "2021-05-08"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-05-15<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [60.0, 4.0, 59.0, 33.0, 277.0, 87.0, 36.0, 15.0, 8.0, 356.0, 137.0, 3.0, 3.0, 179.0, 66.0, 15.0, 22.0, 70.0, 45.0, 6.0, 71.0, 66.0, 413.0, 62.0, 26.0, 36.0, 6.0, 33.0, 32.0, 20.0, 161.0, 15.0, 503.0, 82.0, 5.0, 100.0, 46.0, 60.0, 192.0, 10.0, 52.0, 12.0, 55.0, 305.0, 35.0, 3.0, 123.0, 58.0, 31.0, 54.0, 2.0, 4150.0], "type": "choropleth"}], "name": "2021-05-15"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-05-22<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [74.0, 15.0, 81.0, 23.0, 278.0, 61.0, 39.0, 8.0, 7.0, 407.0, 156.0, 4.0, 16.0, 210.0, 82.0, 36.0, 19.0, 57.0, 44.0, 18.0, 80.0, 76.0, 350.0, 77.0, 25.0, 147.0, 5.0, -36.0, 22.0, 13.0, 132.0, 18.0, 200.0, 96.0, 3.0, 181.0, 40.0, 31.0, 305.0, 4.0, 36.0, 10.0, 68.0, 261.0, 32.0, 3.0, 89.0, 80.0, 15.0, 31.0, 1.0, 4030.0], "type": "choropleth"}], "name": "2021-05-22"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-05-29<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [34.0, 0.0, 81.0, 14.0, 59.0, 73.0, 26.0, 2.0, 7.0, 311.0, 176.0, 5.0, 10.0, 231.0, 62.0, 19.0, 19.0, 77.0, 54.0, 6.0, 612.0, 50.0, 236.0, 55.0, 31.0, 74.0, 10.0, -12.0, 34.0, 8.0, 75.0, 132.0, 178.0, 97.0, 5.0, 152.0, 373.0, 52.0, 158.0, 4.0, 54.0, 9.0, 61.0, 263.0, 11.0, 0.0, 63.0, 63.0, 20.0, 64.0, 6.0, 4204.0], "type": "choropleth"}], "name": "2021-05-29"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-06-05<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [52.0, 0.0, 72.0, 16.0, 247.0, 37.0, 8.0, 16.0, 4.0, 199.0, 149.0, 3.0, 11.0, 116.0, 70.0, 14.0, 8.0, 309.0, 39.0, 14.0, 41.0, 39.0, 212.0, 43.0, 15.0, 60.0, 19.0, 0.0, 14.0, 3.0, 0.0, 17.0, 137.0, 96.0, 4.0, 119.0, 25.0, 21.0, 162.0, 9.0, 24.0, 12.0, 35.0, 218.0, 10.0, 1.0, 69.0, 71.0, 21.0, 97.0, 1.0, 2979.0], "type": "choropleth"}], "name": "2021-06-05"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-06-12<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [68.0, 4.0, 40.0, 14.0, 258.0, 75.0, 15.0, 2.0, 1.0, 292.0, 172.0, 3.0, 15.0, 187.0, 60.0, 30.0, 22.0, 64.0, 33.0, 6.0, 48.0, 30.0, 190.0, 42.0, 28.0, 56.0, 9.0, 26.0, 22.0, 5.0, 0.0, 17.0, 107.0, 114.0, 4.0, 111.0, 9.0, 40.0, 125.0, 5.0, 24.0, 4.0, 22.0, 223.0, 8.0, 0.0, 64.0, -21.0, 36.0, 56.0, 5.0, 2770.0], "type": "choropleth"}], "name": "2021-06-12"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-06-19<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [40.0, 0.0, 82.0, 14.0, 130.0, 44.0, 9.0, 0.0, 4.0, 290.0, 161.0, 5.0, 6.0, 97.0, 50.0, 12.0, 23.0, 25.0, 40.0, 9.0, 33.0, 23.0, 84.0, 33.0, 18.0, 34.0, 10.0, 3.0, 27.0, 7.0, 198.0, 19.0, 78.0, 75.0, 3.0, 75.0, 29.0, 24.0, 86.0, 2.0, 16.0, 1.0, 18.0, 173.0, 7.0, 0.0, 50.0, 5.0, 14.0, 29.0, 9.0, 2224.0], "type": "choropleth"}], "name": "2021-06-19"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-06-26<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [30.0, 1.0, 81.0, 19.0, 323.0, 38.0, 5.0, 14.0, 0.0, 217.0, 74.0, 4.0, 23.0, 71.0, 27.0, 16.0, 18.0, 24.0, 35.0, 4.0, 30.0, 25.0, 97.0, 42.0, 24.0, 27.0, 11.0, 0.0, 18.0, 4.0, 53.0, 19.0, 39.0, 75.0, 5.0, 115.0, 30.0, 11.0, 83.0, 4.0, 27.0, 3.0, 34.0, 171.0, 25.0, 0.0, 49.0, 82.0, 13.0, 45.0, 6.0, 2191.0], "type": "choropleth"}], "name": "2021-06-26"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-07-03<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [22.0, 3.0, 58.0, 20.0, 250.0, 44.0, 4.0, 2.0, 0.0, 213.0, 50.0, 4.0, 13.0, 59.0, 41.0, 17.0, 12.0, 25.0, 34.0, 2.0, 17.0, 12.0, 50.0, 32.0, 24.0, 82.0, 4.0, 2.0, 30.0, 1.0, 0.0, 9.0, 52.0, 19.0, 0.0, 63.0, 4.0, 20.0, 62.0, 2.0, 11.0, 8.0, 21.0, 150.0, 27.0, 2.0, 31.0, 37.0, 23.0, 35.0, 7.0, 1710.0], "type": "choropleth"}], "name": "2021-07-03"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-07-10<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [44.0, 2.0, 48.0, 35.0, 237.0, 23.0, 0.0, 0.0, 2.0, 190.0, 43.0, 1.0, 7.0, 59.0, 51.0, 7.0, 17.0, 26.0, 24.0, 2.0, 15.0, 15.0, 50.0, 12.0, 20.0, 48.0, 7.0, 1.0, 23.0, 3.0, 80.0, 10.0, 34.0, 49.0, 4.0, 36.0, 18.0, 11.0, 40.0, 0.0, 16.0, 1.0, 25.0, 140.0, 15.0, 0.0, 27.0, 58.0, 9.0, 41.0, 4.0, 1630.0], "type": "choropleth"}], "name": "2021-07-10"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-07-17<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [41.0, 2.0, 91.0, 33.0, 225.0, 47.0, 3.0, 2.0, 3.0, 229.0, 72.0, 4.0, 8.0, 65.0, 26.0, 9.0, 21.0, 29.0, 47.0, 20.0, 18.0, 8.0, 50.0, 13.0, 26.0, 87.0, 18.0, 13.0, 38.0, 6.0, 0.0, 25.0, 35.0, 40.0, 5.0, 57.0, 11.0, 25.0, 45.0, 6.0, 16.0, 1.0, 32.0, 162.0, 20.0, 0.0, 27.0, 45.0, 11.0, 10.0, 9.0, 1836.0], "type": "choropleth"}], "name": "2021-07-17"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-07-24<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [40.0, 3.0, 44.0, 60.0, 144.0, 25.0, 4.0, 1.0, 0.0, 291.0, 56.0, 4.0, 10.0, 47.0, 25.0, 0.0, 42.0, 28.0, 72.0, 15.0, 23.0, 26.0, 37.0, 15.0, 37.0, 83.0, 4.0, 9.0, 59.0, 4.0, 55.0, 18.0, 42.0, 47.0, 1.0, 30.0, 37.0, 19.0, 33.0, 3.0, 16.0, 1.0, 45.0, 189.0, 12.0, 1.0, 23.0, 36.0, 17.0, 32.0, 6.0, 1871.0], "type": "choropleth"}], "name": "2021-07-24"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-07-31<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [59.0, 5.0, 80.0, 82.0, 196.0, 35.0, 7.0, 2.0, 3.0, 416.0, 62.0, 10.0, 14.0, 49.0, 40.0, 12.0, 16.0, 22.0, 99.0, 3.0, 15.0, 36.0, 42.0, 18.0, 41.0, 110.0, 9.0, -4.0, 95.0, 2.0, 34.0, 13.0, 41.0, 65.0, 1.0, 25.0, 31.0, 22.0, 27.0, 1.0, 26.0, 2.0, 59.0, 255.0, 26.0, 1.0, 32.0, 44.0, 10.0, 41.0, 10.0, 2342.0], "type": "choropleth"}], "name": "2021-07-31"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-08-07<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [58.0, 8.0, 118.0, 146.0, 300.0, 25.0, 3.0, 133.0, 0.0, 703.0, 126.0, 3.0, 13.0, 76.0, 49.0, 23.0, 44.0, 38.0, 211.0, 1.0, 27.0, 13.0, 33.0, 26.0, 78.0, 160.0, 18.0, 10.0, 93.0, 2.0, 27.0, 15.0, 60.0, 101.0, 3.0, 64.0, 46.0, 31.0, 52.0, 3.0, 46.0, 7.0, 95.0, 350.0, 43.0, 0.0, 26.0, 46.0, 19.0, 9.0, 10.0, 3591.0], "type": "choropleth"}], "name": "2021-08-07"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-08-14<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [165.0, 2.0, 93.0, 163.0, -155.0, 62.0, 11.0, 4.0, 1.0, 770.0, 176.0, 7.0, 35.0, 100.0, 74.0, 17.0, 86.0, 54.0, 309.0, 3.0, 31.0, 47.0, 63.0, 35.0, 140.0, 200.0, 8.0, -5.0, 155.0, 6.0, 0.0, 21.0, 120.0, 90.0, 3.0, 58.0, 63.0, 46.0, 64.0, 1.0, 89.0, 1.0, 114.0, 613.0, 31.0, 4.0, 41.0, 71.0, 11.0, 23.0, 7.0, 4128.0], "type": "choropleth"}], "name": "2021-08-14"}, {"data": [{"coloraxis": "coloraxis", "geo": "geo", "hoverlabel": {"namelength": 0}, "hovertemplate": "end_date=2021-08-21<br>country=%{location}<br>covid_deaths=%{z}", "locationmode": "country names", "locations": ["United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States", "United States"], "name": "", "z": [34.0, 0.0, 29.0, 66.0, 104.0, 9.0, 0.0, 1.0, 0.0, 0.0, 36.0, 5.0, 6.0, 31.0, 16.0, 0.0, 9.0, 25.0, 65.0, 0.0, 3.0, 5.0, 20.0, 5.0, 52.0, 1.0, 6.0, 0.0, 30.0, 0.0, 0.0, 5.0, 42.0, 54.0, 0.0, 0.0, 0.0, 14.0, 0.0, 0.0, 69.0, 0.0, 45.0, 52.0, 12.0, 0.0, 19.0, 9.0, 2.0, 0.0, 0.0, 881.0], "type": "choropleth"}], "name": "2021-08-21"}]);
                        }).then(function(){

var gd = document.getElementById('36958287-aa3e-41fb-81f8-e2a8da9f0203');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>



```python
corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(12,12))
sns.heatmap(corr, mask=mask, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
```


    
![png](output_39_0.png)
    



```python
# death case over 2 years
df['non-covid death'] = df['total_deaths'] -df['covid_deaths']
sns.pairplot(df, vars = ['total_deaths', 'covid_deaths', 'non_covid_deaths'], hue = 'year')
```




    <seaborn.axisgrid.PairGrid at 0x7fb9d9d3e190>




    
![png](/_posts/covid/output_40_1.png)
    


## Explore Covid Death Data in Germany 


```python
data_de = df[df['region']=='Germany']
data_de.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>region</th>
      <th>region_code</th>
      <th>start_date</th>
      <th>end_date</th>
      <th>days</th>
      <th>year</th>
      <th>week</th>
      <th>population</th>
      <th>total_deaths</th>
      <th>covid_deaths</th>
      <th>expected_deaths</th>
      <th>excess_deaths</th>
      <th>non_covid_deaths</th>
      <th>covid_deaths_per_100k</th>
      <th>excess_deaths_per_100k</th>
      <th>excess_deaths_pct_change</th>
      <th>covid_death_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1463</th>
      <td>Germany</td>
      <td>Germany</td>
      <td>0</td>
      <td>2019-12-30</td>
      <td>2020-01-05</td>
      <td>7</td>
      <td>2020</td>
      <td>1</td>
      <td>83900471</td>
      <td>18883.0</td>
      <td>0.0</td>
      <td>19399.361891</td>
      <td>-516.361891</td>
      <td>18883.0</td>
      <td>0.0</td>
      <td>-0.615446</td>
      <td>-0.026617</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1464</th>
      <td>Germany</td>
      <td>Germany</td>
      <td>0</td>
      <td>2020-01-06</td>
      <td>2020-01-12</td>
      <td>7</td>
      <td>2020</td>
      <td>2</td>
      <td>83900471</td>
      <td>19408.0</td>
      <td>0.0</td>
      <td>19754.528558</td>
      <td>-346.528558</td>
      <td>19408.0</td>
      <td>0.0</td>
      <td>-0.413023</td>
      <td>-0.017542</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1465</th>
      <td>Germany</td>
      <td>Germany</td>
      <td>0</td>
      <td>2020-01-13</td>
      <td>2020-01-19</td>
      <td>7</td>
      <td>2020</td>
      <td>3</td>
      <td>83900471</td>
      <td>18953.0</td>
      <td>0.0</td>
      <td>19675.528558</td>
      <td>-722.528558</td>
      <td>18953.0</td>
      <td>0.0</td>
      <td>-0.861173</td>
      <td>-0.036722</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1466</th>
      <td>Germany</td>
      <td>Germany</td>
      <td>0</td>
      <td>2020-01-20</td>
      <td>2020-01-26</td>
      <td>7</td>
      <td>2020</td>
      <td>4</td>
      <td>83900471</td>
      <td>18827.0</td>
      <td>0.0</td>
      <td>19837.695225</td>
      <td>-1010.695225</td>
      <td>18827.0</td>
      <td>0.0</td>
      <td>-1.204636</td>
      <td>-0.050948</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1467</th>
      <td>Germany</td>
      <td>Germany</td>
      <td>0</td>
      <td>2020-01-27</td>
      <td>2020-02-02</td>
      <td>7</td>
      <td>2020</td>
      <td>5</td>
      <td>83900471</td>
      <td>19774.0</td>
      <td>0.0</td>
      <td>20563.361891</td>
      <td>-789.361891</td>
      <td>19774.0</td>
      <td>0.0</td>
      <td>-0.940831</td>
      <td>-0.038387</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig=make_subplots()
fig.add_trace(go.Scatter(x=data_de['start_date'],y=data_de['total_deaths'],name="total_deaths"))
fig.add_trace(go.Scatter(x=data_de['start_date'],y=data_de['covid_deaths'],name="covid_deaths"))
fig.add_trace(go.Scatter(x=data_de['start_date'],y=data_de['expected_deaths'],name="expected_deaths"))
fig.add_trace(go.Scatter(x=data_de['start_date'],y=data_de['excess_deaths'],name="excess_deaths"))


fig.update_layout(autosize=False,width=900,height=600,title_text="Covid Deaths in Germany")
fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="Number",secondary_y=False)
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="24bd0f6c-4333-4d72-a587-a5b70cada0e3" class="plotly-graph-div" style="height:600px; width:900px;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("24bd0f6c-4333-4d72-a587-a5b70cada0e3")) {
                    Plotly.newPlot(
                        '24bd0f6c-4333-4d72-a587-a5b70cada0e3',
                        [{"name": "total_deaths", "type": "scatter", "x": ["2019-12-30", "2020-01-06", "2020-01-13", "2020-01-20", "2020-01-27", "2020-02-03", "2020-02-10", "2020-02-17", "2020-02-24", "2020-03-02", "2020-03-09", "2020-03-16", "2020-03-23", "2020-03-30", "2020-04-06", "2020-04-13", "2020-04-20", "2020-04-27", "2020-05-04", "2020-05-11", "2020-05-18", "2020-05-25", "2020-06-01", "2020-06-08", "2020-06-15", "2020-06-22", "2020-06-29", "2020-07-06", "2020-07-13", "2020-07-20", "2020-07-27", "2020-08-03", "2020-08-10", "2020-08-17", "2020-08-24", "2020-08-31", "2020-09-07", "2020-09-14", "2020-09-21", "2020-09-28", "2020-10-05", "2020-10-12", "2020-10-19", "2020-10-26", "2020-11-02", "2020-11-09", "2020-11-16", "2020-11-23", "2020-11-30", "2020-12-07", "2020-12-14", "2020-12-21", "2020-12-28", "2021-01-04", "2021-01-11", "2021-01-18", "2021-01-25", "2021-02-01", "2021-02-08", "2021-02-15", "2021-02-22", "2021-03-01", "2021-03-08", "2021-03-15", "2021-03-22", "2021-03-29", "2021-04-05", "2021-04-12", "2021-04-19", "2021-04-26", "2021-05-03", "2021-05-10", "2021-05-17", "2021-05-24", "2021-05-31", "2021-06-07", "2021-06-14", "2021-06-21", "2021-06-28", "2021-07-05", "2021-07-12", "2021-07-19", "2021-07-26", "2021-08-02", "2021-08-09", "2021-08-16", "2021-08-23", "2021-08-30", "2021-09-06", "2021-09-13", "2021-09-20", "2021-09-27", "2021-10-04", "2021-10-11", "2021-10-18"], "y": [18883.0, 19408.0, 18953.0, 18827.0, 19774.0, 19038.0, 19648.0, 18953.0, 19505.0, 19667.0, 19849.0, 19722.0, 19678.0, 20662.0, 20502.0, 19261.0, 18557.0, 17920.0, 17651.0, 17001.0, 17163.0, 16776.0, 17262.0, 16601.0, 16401.0, 17293.0, 16457.0, 16142.0, 16537.0, 16887.0, 17433.0, 17470.0, 19720.0, 17559.0, 16656.0, 16789.0, 17155.0, 17555.0, 17446.0, 17614.0, 17501.0, 17754.0, 18588.0, 18584.0, 19007.0, 19727.0, 20182.0, 21304.0, 22604.0, 24003.0, 24724.0, 25554.0, 25474.0, 24828.0, 24491.0, 24065.0, 22634.0, 21719.0, 20914.0, 20456.0, 18938.0, 18425.0, 18589.0, 18315.0, 18367.0, 18477.0, 18793.0, 19040.0, 19210.0, 19473.0, 19286.0, 18569.0, 17764.0, 17119.0, 18123.0, 17370.0, 18998.0, 17132.0, 16930.0, 16892.0, 17569.0, 17519.0, 17129.0, 16917.0, 17443.0, 17334.0, 17022.0, 17180.0, 18644.0, 17880.0, 17986.0, 17965.0, 18025.0, 18725.0, 19624.0]}, {"name": "covid_deaths", "type": "scatter", "x": ["2019-12-30", "2020-01-06", "2020-01-13", "2020-01-20", "2020-01-27", "2020-02-03", "2020-02-10", "2020-02-17", "2020-02-24", "2020-03-02", "2020-03-09", "2020-03-16", "2020-03-23", "2020-03-30", "2020-04-06", "2020-04-13", "2020-04-20", "2020-04-27", "2020-05-04", "2020-05-11", "2020-05-18", "2020-05-25", "2020-06-01", "2020-06-08", "2020-06-15", "2020-06-22", "2020-06-29", "2020-07-06", "2020-07-13", "2020-07-20", "2020-07-27", "2020-08-03", "2020-08-10", "2020-08-17", "2020-08-24", "2020-08-31", "2020-09-07", "2020-09-14", "2020-09-21", "2020-09-28", "2020-10-05", "2020-10-12", "2020-10-19", "2020-10-26", "2020-11-02", "2020-11-09", "2020-11-16", "2020-11-23", "2020-11-30", "2020-12-07", "2020-12-14", "2020-12-21", "2020-12-28", "2021-01-04", "2021-01-11", "2021-01-18", "2021-01-25", "2021-02-01", "2021-02-08", "2021-02-15", "2021-02-22", "2021-03-01", "2021-03-08", "2021-03-15", "2021-03-22", "2021-03-29", "2021-04-05", "2021-04-12", "2021-04-19", "2021-04-26", "2021-05-03", "2021-05-10", "2021-05-17", "2021-05-24", "2021-05-31", "2021-06-07", "2021-06-14", "2021-06-21", "2021-06-28", "2021-07-05", "2021-07-12", "2021-07-19", "2021-07-26", "2021-08-02", "2021-08-09", "2021-08-16", "2021-08-23", "2021-08-30", "2021-09-06", "2021-09-13", "2021-09-20", "2021-09-27", "2021-10-04", "2021-10-11", "2021-10-18"], "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 83.0, 439.0, 1051.0, 1438.0, 1564.0, 1390.0, 890.0, 703.0, 393.0, 321.0, 257.0, 145.0, 116.0, 94.0, 73.0, 55.0, 48.0, 21.0, 32.0, 30.0, 48.0, 33.0, 40.0, 25.0, 30.0, 24.0, 36.0, 74.0, 69.0, 93.0, 172.0, 264.0, 451.0, 859.0, 1201.0, 1586.0, 2147.0, 2683.0, 3117.0, 4294.0, 3897.0, 4494.0, 6145.0, 5965.0, 5395.0, 4867.0, 4545.0, 3399.0, 2839.0, 2206.0, 1832.0, 1479.0, 1293.0, 1203.0, 1101.0, 1440.0, 1552.0, 1619.0, 1621.0, 1552.0, 1322.0, 1263.0, 1002.0, 818.0, 600.0, 551.0, 368.0, 271.0, 202.0, 129.0, 164.0, 132.0, 125.0, 87.0, 109.0, 159.0, 214.0, 265.0, 352.0, 432.0, 389.0, 415.0, 419.0, 489.0]}, {"name": "expected_deaths", "type": "scatter", "x": ["2019-12-30", "2020-01-06", "2020-01-13", "2020-01-20", "2020-01-27", "2020-02-03", "2020-02-10", "2020-02-17", "2020-02-24", "2020-03-02", "2020-03-09", "2020-03-16", "2020-03-23", "2020-03-30", "2020-04-06", "2020-04-13", "2020-04-20", "2020-04-27", "2020-05-04", "2020-05-11", "2020-05-18", "2020-05-25", "2020-06-01", "2020-06-08", "2020-06-15", "2020-06-22", "2020-06-29", "2020-07-06", "2020-07-13", "2020-07-20", "2020-07-27", "2020-08-03", "2020-08-10", "2020-08-17", "2020-08-24", "2020-08-31", "2020-09-07", "2020-09-14", "2020-09-21", "2020-09-28", "2020-10-05", "2020-10-12", "2020-10-19", "2020-10-26", "2020-11-02", "2020-11-09", "2020-11-16", "2020-11-23", "2020-11-30", "2020-12-07", "2020-12-14", "2020-12-21", "2020-12-28", "2021-01-04", "2021-01-11", "2021-01-18", "2021-01-25", "2021-02-01", "2021-02-08", "2021-02-15", "2021-02-22", "2021-03-01", "2021-03-08", "2021-03-15", "2021-03-22", "2021-03-29", "2021-04-05", "2021-04-12", "2021-04-19", "2021-04-26", "2021-05-03", "2021-05-10", "2021-05-17", "2021-05-24", "2021-05-31", "2021-06-07", "2021-06-14", "2021-06-21", "2021-06-28", "2021-07-05", "2021-07-12", "2021-07-19", "2021-07-26", "2021-08-02", "2021-08-09", "2021-08-16", "2021-08-23", "2021-08-30", "2021-09-06", "2021-09-13", "2021-09-20", "2021-09-27", "2021-10-04", "2021-10-11", "2021-10-18"], "y": [19399.3618913845, 19754.5285580521, 19675.5285580521, 19837.6952247188, 20563.3618913854, 20447.0285580521, 21000.0285580521, 21263.6952247188, 22233.234269662596, 21721.8342696626, 20794.6342696626, 19931.0342696626, 19261.8342696626, 18807.2342696626, 18221.2342696626, 17799.2342696626, 17745.8342696626, 17282.0342696626, 17555.0342696626, 17120.4342696626, 16809.4342696626, 16959.0342696626, 16802.8342696626, 16305.8342696626, 16643.4342696626, 16760.2342696626, 17293.2342696626, 16582.4342696626, 16993.2342696626, 17568.8342696626, 17231.4342696626, 16865.4342696626, 16699.6342696626, 16504.4342696626, 16661.2342696626, 16248.2342696626, 16490.4342696626, 16536.6342696626, 16701.6342696626, 16870.6342696626, 17543.8342696626, 17447.6342696626, 17461.6342696626, 17381.4342696626, 17805.4342696626, 17854.8342696626, 18028.6342696626, 18176.8342696626, 18593.8342696626, 18574.8342696626, 18944.6342696626, 18623.4342696626, 18464.926966291798, 19491.8399812721, 19847.0066479397, 19768.0066479397, 19930.1733146064, 20655.8399812731, 20539.5066479397, 21092.5066479397, 21356.1733146064, 22325.7123595502, 21814.3123595502, 20887.1123595502, 20023.5123595502, 19354.3123595502, 18899.7123595502, 18313.7123595502, 17891.7123595502, 17838.3123595502, 17374.5123595502, 17647.5123595502, 17212.9123595502, 16901.9123595502, 17051.5123595502, 16895.3123595502, 16398.3123595502, 16735.9123595502, 16852.7123595502, 17385.7123595502, 16674.9123595502, 17085.7123595502, 17661.3123595502, 17323.9123595502, 16957.9123595502, 16792.1123595502, 16596.9123595502, 16753.7123595502, 16340.7123595502, 16582.9123595502, 16629.1123595502, 16794.1123595502, 16963.1123595502, 17636.3123595502, 17540.1123595502]}, {"name": "excess_deaths", "type": "scatter", "x": ["2019-12-30", "2020-01-06", "2020-01-13", "2020-01-20", "2020-01-27", "2020-02-03", "2020-02-10", "2020-02-17", "2020-02-24", "2020-03-02", "2020-03-09", "2020-03-16", "2020-03-23", "2020-03-30", "2020-04-06", "2020-04-13", "2020-04-20", "2020-04-27", "2020-05-04", "2020-05-11", "2020-05-18", "2020-05-25", "2020-06-01", "2020-06-08", "2020-06-15", "2020-06-22", "2020-06-29", "2020-07-06", "2020-07-13", "2020-07-20", "2020-07-27", "2020-08-03", "2020-08-10", "2020-08-17", "2020-08-24", "2020-08-31", "2020-09-07", "2020-09-14", "2020-09-21", "2020-09-28", "2020-10-05", "2020-10-12", "2020-10-19", "2020-10-26", "2020-11-02", "2020-11-09", "2020-11-16", "2020-11-23", "2020-11-30", "2020-12-07", "2020-12-14", "2020-12-21", "2020-12-28", "2021-01-04", "2021-01-11", "2021-01-18", "2021-01-25", "2021-02-01", "2021-02-08", "2021-02-15", "2021-02-22", "2021-03-01", "2021-03-08", "2021-03-15", "2021-03-22", "2021-03-29", "2021-04-05", "2021-04-12", "2021-04-19", "2021-04-26", "2021-05-03", "2021-05-10", "2021-05-17", "2021-05-24", "2021-05-31", "2021-06-07", "2021-06-14", "2021-06-21", "2021-06-28", "2021-07-05", "2021-07-12", "2021-07-19", "2021-07-26", "2021-08-02", "2021-08-09", "2021-08-16", "2021-08-23", "2021-08-30", "2021-09-06", "2021-09-13", "2021-09-20", "2021-09-27", "2021-10-04", "2021-10-11", "2021-10-18"], "y": [-516.361891384491, -346.52855805209, -722.528558052094, -1010.69522471877, -789.361891385426, -1409.0285580520901, -1352.02855805209, -2310.69522471876, -2728.23426966258, -2054.83426966258, -945.6342696625801, -209.034269662588, 416.16573033742304, 1854.7657303374103, 2280.7657303374303, 1461.76573033742, 811.1657303374159, 637.965730337426, 95.96573033742241, -119.434269662572, 353.565730337417, -183.034269662585, 459.165730337416, 295.165730337418, -242.434269662594, 532.765730337414, -836.234269662578, -440.43426966259005, -456.234269662571, -681.834269662584, 201.565730337414, 604.5657303374211, 3020.3657303374202, 1054.56573033743, -5.2342696625855805, 540.76573033742, 664.56573033741, 1018.3657303374199, 744.365730337428, 743.365730337417, -42.834269662580496, 306.36573033742, 1126.3657303374198, 1202.5657303374198, 1201.5657303374198, 1872.16573033742, 2153.3657303374202, 3127.16573033742, 4010.16573033742, 5428.16573033742, 5779.36573033742, 6930.565730337421, 7009.0730337082105, 5336.160018727871, 4643.993352060271, 4296.99335206026, 2703.8266853935897, 1063.16001872693, 374.493352060264, -636.506647939736, -2418.1733146064003, -3900.7123595502203, -3225.3123595502198, -2572.11235955022, -1656.51235955023, -877.312359550218, -106.71235955022699, 726.2876404497839, 1318.2876404497802, 1634.6876404497698, 1911.48764044978, 921.487640449781, 551.0876404497869, 217.087640449776, 1071.48764044977, 474.687640449774, 2599.68764044978, 396.087640449765, 77.28764044977291, -493.71235955022, 894.087640449769, 433.287640449787, -532.312359550226, -406.91235955022796, 485.087640449779, 541.8876404497751, 425.08764044978994, 426.28764044977305, 2303.28764044978, 1297.0876404497699, 1356.88764044978, 1170.88764044979, 1061.88764044978, 1088.68764044978, 2083.88764044978]}],
                        {"autosize": false, "height": 600, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Covid Deaths in Germany"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "Date"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "Number"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('24bd0f6c-4333-4d72-a587-a5b70cada0e3');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


We see that excess death is close to covid death


```python
plt.figure(figsize=(12,8))
df_temp = data_de['end_date'].str.split('-', expand=True)[[1,0]]
data_de['date'] = df_temp[1] + '/' + df_temp[0]
sns.barplot(data=data_de, x='total_deaths', y='date', color='orange', label='Total Deaths')
sns.barplot(data=data_de, x='covid_deaths', y='date', color='grey', label='Covid Deaths')
plt.xlabel(xlabel = 'Number of Deaths',fontsize=16, fontweight='bold')
plt.ylabel(ylabel = 'Date',fontsize=16, fontweight='bold')
plt.legend()
plt.show()
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:3: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
    


    
![png](/_posts/covid/output_45_1.png)
    



```python

```
