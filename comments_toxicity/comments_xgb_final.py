import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


def optimal_n_rounds(xgb_model, xgb_matrix, max_n_estimators):
	""" take the input model and xgb matrix (x and y values) 
		and determine the optimal number of trees via cross validation.
		 returns the number of trees """
	cvresult = xgb.cv(xgb_model, x_values, 
						num_boost_round = max_n_estimators, 
						nfold = 5,
						metrics	= 'auc', 
						early_stopping_rounds = 50)	
	return cvresult.shape[0]

def optimal_params(xgb_model, x_vals, y_vals, xgb_param_grid):
	""" take a model, predictor matrix and paramater grid and
		return the optimal paramater set """
	_gsearch = GridSearchCV(xgb_model,  xgb_param_grid, 
								scoring='roc_auc', 
								n_jobs=4, 
								iid=False, 
								cv=3)
	_gsearch.fit(x_vals, y_vals)

	return _gsearch.best_params_


if __name__ == '__main__':

	#load in the processed data from train_and_test_to_matrix.py
	train_sparse = sparse.load_npz('sparse_train_punc.npz')
	test_sparse = sparse.load_npz('sparse_test_punc.npz')

	train = pd.read_csv('train.csv')
	test = pd.read_csv('test.csv')
	sub_file = pd.read_csv('sample_submission.csv')


	to_predict = list(train.columns[2:])

	for col in to_predict:

		xgtrain_input = xgb.DMatrix(train_sparse, label=train[col].values)

		xgb_initial = xgb.XGBClassifier(learning_rate =0.1,
									n_estimators=1000,
									max_depth=5,
									min_child_weight=1,
									gamma=0,
									subsample=0.8,
									colsample_bytree=0.8,
									objective= 'binary:logistic',
									scale_pos_weight=1)

		opt_rounds = optimal_n_rounds(xgb_initial, xgtrain_input, 1000)

		xgb_class_grid = xgb.XGBClassifier(n_estimators=opt_rounds,
											gamma=0,
											subsample=0.8,
											objective= 'binary:logistic',
											scale_pos_weight=1)
		
		xgb_params = {'max_depth':[4, 6, 8],
						'min_child_weight':[1, 4, 8],
						'colsample_bytree': [0.8, 0.9], }


		xgb_best = optimal_params(xgb_class_grid, 
									train_sparse, 
									train[col].values, 
									xgb_params)

		xgb_final = xgb.XGBClassifier(gsearch_toxic.best_params_,
								eta = 0.001,
								n_estimators=5000,
								gamma=0,
								subsample=0.8,
								objective= 'binary:logistic',
								scale_pos_weight=1,
								early_stopping_rounds = 50)

		xgb_final.fit(train_sparse, train[col].values, eval_metric='auc')
		
		optimal_predictions = xgb_final.predict(test_sparse)

		sub_file[col] = optimal_predictions


	sub_file.to_csv('cam_xgb_predictions.csv', index = False)





