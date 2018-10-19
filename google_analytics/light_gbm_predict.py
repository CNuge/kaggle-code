import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn.model_selection import train_test_split

#run
X_train = np.load('X_train.dat')
y_train = np.load('y_train.dat')
X_test = np.load('X_test.dat')

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
	"learning_rate" : 0.01,
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
                  early_stopping_rounds = 100,
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


def set_min_zero(x):
	if x < 0:
		return 0
	else:
		return x

submission['PredictedLogRevenue'] = submission['PredictedLogRevenue'].apply(
									lambda x: set_min_zero(x))


#submit the output
submission.to_csv('cam_lightgbm_pred3_floor.csv', index = False)
#1.78 first go, worse than all 0s
#1.775 on second... beating the all 0s but barely.
#1.6371 on third... making gains now


"""
changes:
try3 : dropped the categoricals with 50+ options, possibly too much noise in the features
training
[1968]	training's rmse: 1.53077	valid_1's rmse: 1.6381
lb: 1.6371


NOTES:
#train, drop features/ repeat seems like a good way to go... removing the noisy columns
greatly improved the accuracy


#need to figure out what is causing the lack of strength in predictions.
#try:
#upsampling the #s
#PCA
#increase the train size?



#something is still wrong here... shouldn't be getting negative predictions
#need to find out how to ensure the predictions stay positive

#once the above is working try the following:
1. up the iterations to train a little 
2. grid search to pick better hyperparams
3. need a faster learning rate to get things working
"""