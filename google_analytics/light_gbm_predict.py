import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn.model_selection import train_test_split


X_train.shape 
y_train.shape 
X_test.shape 

#split off a validation set
X_trainr, X_val, y_trainr, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 1738)


#basic starting params, can tune via grid search once things moving well
lgb_params = {
	"seed": 1738,
	"bagging_seed" : 42,
	"objective" : "regression",
	"metric" : "rmse",
	"num_leaves" : 40,
	"learning_rate" : 0.001,
	"bagging_fraction" : 0.6,
	"feature_fraction" : 0.6,
	}

#load in lgbm matrix fmt
ltrain = lgb.Dataset(X_trainr, label = y_trainr)
lval = lgb.Dataset(X_val, label = y_val)

#build and train the model
lgb_model1 = lgb.train(lgb_params, ltrain, 
                  num_boost_round = 50000,
                  valid_sets = [ltrain, lval],
                  early_stopping_rounds = 500,
                  verbose_eval = 100)


#make predictions on the test data
test_y = lgb_model1.predict(X_test, num_iteration = lgb_model1.best_iteration)

# sum the predictions using the defined formula to get a revenue by user metric
# aggregate on 'fullVisitorId' 
# final_test['fullVisitorId' ]


final_pred = final_test[['fullVisitorId']].copy()

final_pred['train_yht'] = test_y

"""
########
# experiment - try without this as well see if setting the floor is causing the issue
def set_floor(x):
	if x < 0:
		return 0
	else:
		return x

final_pred['train_yht'] = final_pred['train_yht'].apply(lambda x: set_floor(x))
####
"""

#issue - the ids in the submission file and the ids in the test aren't a 1:1 match?
#have I jumbled them or something?


final_by_ind =  final_pred.groupby(['fullVisitorId']).sum()

final_by_ind = final_by_ind.reset_index()

final_by_ind.head()

submission = submission.merge(final_by_ind, on = 'fullVisitorId', how = 'left')

submission['PredictedLogRevenue'] = np.log1p(submission['train_yht'])

submission['PredictedLogRevenue'] = submission['PredictedLogRevenue'].fillna(0)

submission = submission.drop(['train_yht'], axis = 1)
submission.head()

#submit
submission.to_csv('cam_pred1.csv', index = False)

"""
#once the above is working try the following:
1. up the iterations to train a little 
2. grid search to pick better hyperparams
3. 
"""