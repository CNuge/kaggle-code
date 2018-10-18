import pandas as pd
from pandas import Series, DataFrame
import numpy as np

import json
import gc

print('loading: test')
test = pd.read_csv('./data/test.csv')

test.head()

#which columns have json
#device
json_cols = ['device', 'geoNetwork', 'totals',  'trafficSource']
column = 'device'

for column in json_cols:
	print(f'cleaning: {column}')
	c_load = test[column].apply(json.loads)
	c_list = list(c_load)
	c_dat = json.dumps(c_list)

	intermediate = pd.read_json(c_dat)

	if 'fullVisitorId' in intermediate.columns:
		intermediate = intermediate.drop(['fullVisitorId'], axis=1)

	print('merging to main')
	test = test.join(intermediate)
	test = test.drop(column , axis=1)

print('writing test data')
test.head()
test.to_csv('./data/test_cleaned.csv', index=False)


print('loading: train')
train = pd.read_csv('./data/train.csv')
train.head()

test = []
gc.collect()
#which columns have json
#device
json_cols = ['device', 'geoNetwork', 'totals',  'trafficSource', 'adwordsClickInfo']
column = 'device'

for column in json_cols:
	print(f'cleaning: {column}')

	c_load = train[column].apply(json.loads)
	c_list = list(c_load)
	c_dat = json.dumps(c_list)
	
	intermediate = pd.read_json(c_dat)

	if 'fullVisitorId' in intermediate.columns:
		intermediate = intermediate.drop(['fullVisitorId'], axis=1)


	print('merging to main')
	train = train.join(intermediate)
	train = train.drop(column , axis=1)

print('writing train data')
train.head()
train.to_csv('./data/train_cleaned.csv', index=False)
