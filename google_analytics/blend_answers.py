"""
take the predictions from the different models I have produced and merge them into
a single submission file, weighting the evenly at first and then adjusting.
"""
import pandas as pd
import os
import numpy as np


weights = { 
	'cam_lightgbm_pred3_floor.csv' : ,
	'cam_xgb_pred1.csv' : ,
	}

data = {}

for mod in weights.keys():
	data[mod] = pd.read_csv(mod)

for k, v in data.items():
	print(k)
	print(v.columns)

test_ids = pd.read_csv('sample_submission.csv')

revenue = ['PredictedLogRevenue']

for col in revenue:
	preds = [data[x][col] * weights[x] for x in data.keys()]
	pred = preds[0]
	for i in preds[1:]:
		pred = pred + i
	test_ids[col] = [float(format(x, '.6f')) for x in pred]


test_ids.to_csv('cam_blended_submission.csv', index=False)



