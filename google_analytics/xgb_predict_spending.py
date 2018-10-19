import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.model_selection import train_test_split


X_train = np.load('X_train.dat')
y_train = np.load('y_train.dat')
X_test = np.load('X_test.dat')


y_median = np.median(y_train) 
y_mean = np.mean(y_train)


X_trainr, X_val, y_trainr, y_val = train_test_split(X_train, y_train, 
									test_size = 0.2, random_state = 38)


xgb_train = xgb.DMatrix(X_trainr, y_trainr)
xgb_valid = xgb.DMatrix(X_val, y_val)
xgb_test = xgb.DMatrix(X_test)

y_mean = np.mean(y_train) 
y_median = np.median(y_train)


xgb_params = {'n_estimators': 500,
				'eta' :  0.05,
                'max_depth' :  8,
                'subsample' : 0.80, 
                'objective' :  'reg:linear',
                'eval_metric' : 'rmse',
                'base_score' :  y_mean,
                'silent': 1}

watchlist = [(xgb_train, 'train'), (xgb_valid, 'valid')]

xbg_model1 = xgb.train(xgb_params, dtrain, 
                num_boost_round = 500,
                evals = watchlist,
				early_stopping_rounds=25, 
				verbose_eval=25)

print(f'Best score: RMSE {xbg_model1.best_score}')


#make predictions on the test data
test_y = xbg_model1.predict(xgb_test)

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
submission.to_csv('cam_xgb_pred1.csv', index = False)

"""
########################################################

cv_result = xgb.cv(xgb_params, dtrain, 
					nfold=5, 
					num_boost_round=2000, 
					early_stopping_rounds=50, 
					verbose_eval=0, show_stdv=False)

num_boost_rounds = len(cv_result)

#traing the model on the full all_train dataset
model = xgb.train(xgb_params, dtrain, 
                  num_boost_round = num_boost_rounds)




#####################################################################
"""
