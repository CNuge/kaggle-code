import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

X_train = np.load('X_train.dat')
y_train = np.load('y_train.dat')
X_test = np.load('X_test.dat')


"""
run a PCA on the features, take the x number of features explaining 95%
of the variance and train up the model using the PCs for the train/test matricies

#the first axis is explining 100% of the variation in the data?
- need to trim down the number of features we have first and then run the pca




- come back to this after finishing valid_reduce.py

"""

dim_red = PCA(n_components = 10)

PC_X_train = dim_red.fit_transform(X_train)
PC_X_test = dim_red.transform(X_test)

PC_X_train.shape
PC_X_train[:5]
PC_X_test.shape
PC_X_test[:5]

dim_red.n_components_
dim_red.components_
dim_red.explained_variance_
dim_red.explained_variance_ratio_

y_median = np.median(y_train) 
y_mean = np.mean(y_train)


X_trainr, X_val, y_trainr, y_val = train_test_split(PC_X_train, y_train, 
									test_size = 0.2, random_state = 38)


xgb_train = xgb.DMatrix(X_trainr, y_trainr)
xgb_valid = xgb.DMatrix(X_val, y_val)
xgb_test = xgb.DMatrix(X_test)

y_mean = np.mean(y_train) 
y_median = np.median(y_train)


xgb_params = {'n_estimators': 5000,
				'eta' :  0.05,
                'max_depth' :  8,
                'subsample' : 0.80, 
                'objective' :  'reg:linear',
                'eval_metric' : 'rmse',
                'base_score' :  y_mean,
                'silent': 1}

watchlist = [(xgb_train, 'train'), (xgb_valid, 'valid')]

xbg_model1 = xgb.train(xgb_params, xgb_train, 
                num_boost_round = 500,
                evals = watchlist,
				early_stopping_rounds=50, 
				verbose_eval=25)

print(f'Best score: validation RMSE = {xbg_model1.best_score}')


#make predictions on the test data
test_y = xbg_model1.predict(PC_X_test)

final_test = pd.read_csv('./data/test_cleaned.csv')
final_test['fullVisitorId'] = final_test['fullVisitorId'].astype('str')

#group by id

final_pred = final_test[['fullVisitorId']].copy()

final_pred['test_pred'] = test_y

final_by_ind =  final_pred.groupby(['fullVisitorId']).sum()


#move index to a col
final_by_ind = final_by_ind.reset_index()

#merge the predictions with the sample sub

submission = pd.read_csv('./data/sample_submission.csv')
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
submission.to_csv('cam_xgb_pred_pca.csv', index = False)

