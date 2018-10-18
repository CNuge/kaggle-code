import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn.model_selection import train_test_split


X_train.shape 
y_train.shape 
X_test.shape 

#split off a validation set
X_trainr, X_val, y_trainr, y_val = train_test_split(X, y, test_size=0.2, random_state=1738)

#basic starting params, can tune via grid search once things moving well
lgb_params = {
	"seed": 1738,
	"objective" : "regression",
	"metric" : "rmse",
	"num_leaves" : 40,
	"learning_rate" : 0.01,
	"bagging_fraction" : 0.5,
	"feature_fraction" : 0.5,
	"bagging_frequency" : 6,
	"bagging_seed" : 42,
	}

#load in lgbm matrix fmt
ltrain = lgb.Dataset(X_trainr, label=y_trainr)
lval = lgb.Dataset(X_val, label=y_val)

#build and train the model
lgb_model1 = lgb.train(params, lgb_train_data, 
                  num_boost_round=2000,
                  valid_sets=[lgb_train_data, lgb_val_data],
                  early_stopping_rounds=50,
                  verbose_eval=100)


#make predictions on the test data
test_y = lgb_model1.predict(X_test, num_iteration = lgb_model1.best_iteration)

# sum the predictions using the defined formula to get a revenue by user metric
# aggregate on 'fullVisitorId' 
# final_test['fullVisitorId' ]

final_pred = final_test['fullVisitorId']

final_pred[train_yht] = test_y


final_pred = final_pred.sort(['fullVisitorId'])

final_by_ind =  final_pred.groupby(['fullVisitorId']).sum()

final_by_ind = final_by_ind.add_suffix('_sum').reset_index()

final_by_ind['PredictedLogRevenue'] = np.log1p(final_by_ind['train_yht_sum'])

#submit
final_by_ind.to_csv('cam_pred1.csv')

"""
#once the above is working try the following:
1. up the iterations to train a little 
2. grid search to pick better hyperparams
3. 
"""