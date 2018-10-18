import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn.model_selection import train_test_split


X_train.shape 
y_train.shape

X_test.shape 

#split off a validation set
X_trainr, X_val, y_trainr, y_val = train_test_split(X_train, y_train, 
									test_size = 0.2, random_state = 1738)


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

final_pred = final_test[['fullVisitorId']].copy()

final_pred['test_pred'] = test_y

# sum the predictions using the defined formula to get a revenue by user metric
# aggregate on 'fullVisitorId' 
# final_test['fullVisitorId' ]
#issue - the ids in the submission file and the ids in the test aren't a 1:1 match?
#have I jumbled them or something?
#resolved - they were mixed type in the train, some string, some int... 
# I flipped all ids in both sub and test to str... still not all there :/


#group by id
final_by_ind =  final_pred.groupby(['fullVisitorId']).sum()
#move index to a col
final_by_ind = final_by_ind.reset_index()

#merge the predictions with the sample sub
submission = submission.merge(final_by_ind, on = 'fullVisitorId', how = 'left')
#fill nas and move to right column name
submission['PredictedLogRevenue'] = submission['test_pred'].fillna(0.0)
submission = submission.drop(['test_pred'], axis = 1)

#submit the output
submission.to_csv('cam_lightgbm_pred1.csv', index = False)
#1.78 first go


"""
#once the above is working try the following:
1. up the iterations to train a little 
2. grid search to pick better hyperparams
3. 
"""