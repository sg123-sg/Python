# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 23:46:31 2021

@author: Sourav
"""

from alpha_vantage.timeseries import TimeSeries
import json


def save_dataset(symbol):
    credentials = json.load(open('creds.json', 'r'))
    api_key = credentials['MSW7WDGI7TWER7T1']

    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol, outputsize='full')

    data.to_csv(f'./{TSLA}_daily.csv')
import pandas as pd
from sklearn import preprocessing
import numpy as np

history_points = 50

def csv_to_dataset(csv_path):
    data = pd.read_csv(csv_path)
    data = data.drop('date', axis=1)
    data = data.drop(0, axis=0)
    
data_normaliser = preprocessing.MinMaxScaler()
data_normalised = data_normaliser.fit_transform(data)

# using the last {history_points} open high low close volume data points, predict the next open value
ohlcv_histories_normalised =      np.array([data_normalised[i  : i + history_points].copy() for i in range(len(data_normalised) - history_points)])
next_day_open_values_normalised = np.array([data_normalised[:,0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])
next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

next_day_open_values = np.array([data[:,0][i + history_points].copy() for i in range(len(data) - history_points)])
next_day_open_values = np.expand_dims(next_day_open_values_normalised, -1)

y_normaliser = preprocessing.MinMaxScaler()
y_normaliser.fit(np.expand_dims( next_day_open_values ))
    
    assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0]
    return ohlcv_histories_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser

test_split = 0.9 # the percent of data to be used for testing
n = int(ohlcv_histories.shape[0] * test_split)

# splitting the dataset up into train and test sets

ohlcv_train = ohlcv_histories[:n]
y_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
y_test = next_day_open_values[n:]

unscaled_y_test = unscaled_y[n:]

ohlcv_histories, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('MSFT_daily.csv')
    
    
