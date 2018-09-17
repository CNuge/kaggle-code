import pandas as pd
from pandas import Series, DataFrame
import numpy as np

import json
import gc

test = pd.read_csv('test.csv')

test.head()

#which columns have json
#device
json_cols = ['device', 'geoNetwork', 'totals',  'trafficSource']
column = 'device'

for column in json_cols:

	c_load = test[column].apply(json.loads)
	c_list = list(c_load)
	c_dat = json.dumps(c_list)

	test = test.join(pd.read_json(c_dat))
	test = test.drop(column , axis=1)

test.head()
test.to_csv('test_cleaned.csv')

train = pd.read_csv('train.csv')
train.head()

test = []
gc.collect()
#which columns have json
#device
json_cols = ['device', 'geoNetwork', 'totals',  'trafficSource']
column = 'device'

for column in json_cols:

	c_load = train[column].apply(json.loads)
	c_list = list(c_load)
	c_dat = json.dumps(c_list)

	train = train.join(pd.read_json(c_dat))
	train = train.drop(column , axis=1)

train.head()
train.to_csv('train_cleaned.csv')
