# -*- coding: utf-8 -*-
"""Inventory Demand Forecasting using Machine Learning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d52rfPIxUH9xPtJSjVCBbsPlHQ9I5jbV
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/content/train.csv')
display(df.head())
display(df.tail())

df.shape

df.info()

df.describe()

parts = df["date"].str.split("-", n = 3, expand = True)
df["year"]= parts[0].astype('int')
df["month"]= parts[1].astype('int')
df["day"]= parts[2].astype('int')
df.head()

from datetime import datetime
import calendar

def weekend_or_weekday(year,month,day):

	d = datetime(year,month,day)
	if d.weekday()>4:
		return 1
	else:
		return 0

df['weekend'] = df.apply(lambda x:weekend_or_weekday(x['year'], x['month'], x['day']), axis=1)
df.head()

from datetime import date
import holidays
import pandas as pd

# Create a DataFrame with date column
df = pd.DataFrame({'date': pd.date_range(start='2023-01-01', end='2023-12-31')})

# Define a function to check holidays for a batch of dates
def is_holiday_batch(start_date, end_date):
    india_holidays = holidays.country_holidays('IN')
    df['holidays'] = df['date'].apply(lambda x: 1 if india_holidays.get(x) else 0)

# Process the DataFrame in batches
batch_size = 100
for start in range(0, len(df), batch_size):
    end = start + batch_size
    is_holiday_batch(df['date'].iloc[start:end], df['date'].iloc[start:end])

df.head()

df['m1'] = np.sin(df['month'] * (2 * np.pi / 12))
df['m2'] = np.cos(df['month'] * (2 * np.pi / 12))
df.head()

def which_day(year, month, day):

	d = datetime(year,month,day)
	return d.weekday()

df['weekday'] = df.apply(lambda x: which_day(x['year'],
													x['month'],
													x['day']),
								axis=1)
df.head()

df.drop('date', axis=1, inplace=True)

df['store'].nunique(), df['item'].nunique()

import matplotlib.pyplot as plt

features = ['store', 'year', 'month', 'weekday', 'weekend']

# Create a grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# Flatten the 2D axes array for easier indexing
axes = axes.flatten()

for i, col in enumerate(features):
    if col in df:
        df.groupby(col).mean()['sales'].plot.bar(ax=axes[i])
        axes[i].set_title(f'Mean Sales by {col}')

# Remove any extra empty subplots
for j in range(len(features), 6):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

plt.figure(figsize=(10,5))
df.groupby('day').mean()['sales'].plot()
plt.show()

plt.figure(figsize=(10,5))
df.groupby('day').mean()['sales'].plot()
plt.show()

plt.figure(figsize=(15, 10))

# Calculating Simple Moving Average
# for a window period of 30 days
window_size = 30
data = df[df['year']==2013]
windows = data['sales'].rolling(window_size)
sma = windows.mean()
sma = sma[window_size - 1:]

data['sales'].plot()
sma.plot()
plt.legend()
plt.show()

plt.subplots(figsize=(12, 5))
plt.subplot(1, 2, 1)
sb.distplot(df['sales'])

plt.subplot(1, 2, 2)
sb.boxplot(df['sales'])
plt.show()

plt.figure(figsize=(10, 10))
sb.heatmap(df.corr() > 0.8,
		annot=True,
		cbar=False)
plt.show()

df = df[df['sales']<140]

features = df.drop(['sales', 'year'], axis=1)
target = df['sales'].values


X_train, X_val, Y_train, Y_val = train_test_split(features, target,
												test_size = 0.05,
												random_state=22)
X_train.shape, X_val.shape

# Normalizing the features for stable and fast training.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

models = [LinearRegression(), XGBRegressor(), Lasso(), Ridge()]

for i in range(4):
	models[i].fit(X_train, Y_train)

	print(f'{models[i]} : ')

	train_preds = models[i].predict(X_train)
	print('Training Error : ', mae(Y_train, train_preds))

	val_preds = models[i].predict(X_val)
	print('Validation Error : ', mae(Y_val, val_preds))
	print()






