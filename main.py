"""
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

train_df= pd.read_csv('../input/gdz-elektrik-datathon/train.csv')
test_df= pd.read_csv('../input/gdz-elektrik-datathon/test.csv')
holidays_df = pd.read_csv('../input/gdz-elektrik-datathon/holidays.csv')
weather_df= pd.read_csv('../input/gdz-elektrik-datathon/weather.csv')

print(train_df.head())
print("----------------")
print(train_df.dtypes)

train_df['tarih'] = pd.to_datetime(train_df['tarih'])
train_df['ilce'] = train_df['ilce'].astype('category')
train_df["bildirimsiz_sum"] = train_df["bildirimsiz_sum"].astype(np.int8)
train_df["bildirimli_sum"] = train_df["bildirimli_sum"].astype(np.int8)

test_df['tarih'] = pd.to_datetime(test_df['tarih'])
test_df['ilce'] = test_df['ilce'].astype('category')
test_df["bildirimli_sum"] = test_df["bildirimli_sum"].astype(np.int8)

print(train_df.head())
print("----------------")
print(train_df.dtypes)

print(test_df.head())
print("----------------")
print(test_df.dtypes)

print(holidays_df.head())
print(holidays_df.columns)

holidays_df["tarih"] = holidays_df['Yıl'].astype(str) + '-' + holidays_df['Ay'].astype(str) + '-' + holidays_df['Gün'].astype(str)
holidays_df["tarih"] = pd.to_datetime(holidays_df["tarih"])
holidays_df = holidays_df.drop(columns=['Yıl', 'Ay', 'Gün'])

print(holidays_df.dtypes)
print(holidays_df.head())
print(weather_df.dtypes)
print("-----------------")
print(weather_df.head())

weather_df["tarih"] = pd.to_datetime(weather_df["date"])
weather_df = weather_df.drop(columns=['date'])
weather_df['ilce'] = weather_df['ilce'].astype('category')

print(weather_df.dtypes)
print("-----------------")
print(weather_df.head())

print("Weather:")
print(weather_df.isnull().any())
print("----------------------\n Test:")
print(test_df.isnull().any())
print("----------------------\n Train:")
print(train_df.isnull().any())
print("----------------------\n Holidays:")
print(holidays_df.isnull().any())

print("Weather:")
print(weather_df.isna().any())
print("----------------------\n Test:")
print(test_df.isna().any())
print("----------------------\n Train:")
print(train_df.isna().any())
print("----------------------\n Holidays:")
print(holidays_df.isna().any())

print(weather_df.columns)

max_columns = ['t_2m:C', 'effective_cloud_cover:p', 'global_rad:W', 'relative_humidity_2m:p', 'wind_speed_10m:ms', 'prob_precip_1h:p', 't_apparent:C']
min_columns = ['t_2m:C', 'effective_cloud_cover:p', 'global_rad:W', 'relative_humidity_2m:p', 'wind_speed_10m:ms', 'prob_precip_1h:p', 't_apparent:C']
mode_columns = ['t_2m:C', 'effective_cloud_cover:p', 'global_rad:W', 'relative_humidity_2m:p', 'wind_speed_10m:ms', 'prob_precip_1h:p', 't_apparent:C']

for col in max_columns:
    weather_df[col + '_max'] = weather_df.groupby(weather_df['tarih'].dt.date)[col].transform('max')

for col in min_columns:
    weather_df[col + '_min'] = weather_df.groupby(weather_df['tarih'].dt.date)[col].transform('min')

for col in mode_columns:
    weather_df[col + '_mode'] = weather_df.groupby(weather_df['tarih'].dt.date)[col].transform(lambda x: x.mode()[0])

print(weather_df.head())
print(weather_df.columns)

weather_df = weather_df.drop(columns=max_columns)
print(weather_df.head())
print(weather_df.columns)

print(weather_df.isnull().any())
print("------------------------")
print(weather_df.isna().any())

print("train_df:",train_df.columns)
print("weather_df:", weather_df.columns)
print("holidays_df:", holidays_df.columns)


merged_df = pd.merge(train_df, weather_df, on=['tarih', 'ilce'], how='inner')

print(merged_df.head())
print(merged_df.columns)
print(merged_df.isnull().any())
print("------------")
print(merged_df.isna().any())

merged_df = pd.merge(merged_df, holidays_df, on='tarih', how='left')
merged_df['Bayram_Flag'].fillna(0, inplace=True)

print(merged_df.dtypes)
print(merged_df.head())
"""


