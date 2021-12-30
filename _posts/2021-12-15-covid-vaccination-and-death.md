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

{% include_relative covid/fully_vaccinated.html %}

## Vaccination in Germany over time


```python
df_de = df[df['iso_code'] == 'DEU'].sort_values('date')
df_de.head()
```




<div>

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


{% include_relative covid/de_fully_vaccinated.html %}


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

{% include_relative covid/covid_death.html %}


```python
corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(12,12))
sns.heatmap(corr, mask=mask, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
```


    
![png](/_posts/covid/output_39_0.png)
    



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

{% include_relative covid/de_covid_death.html %}


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


    
![png](/_posts/covid/output_45_1.png)
    



