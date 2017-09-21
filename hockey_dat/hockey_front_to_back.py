import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Imputer
from sklearn.pipeline import FeatureUnion
from datetime import datetime
import gc

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error



train = pd.read_csv('train.csv', encoding = "ISO-8859-1")
test_x = pd.read_csv('test.csv', encoding = "ISO-8859-1")
test_y = pd.read_csv('test_salaries.csv') 

test_y=list(test_y['Salary'].values)

train_x = train.drop('Salary',axis=1)
train_y = list(train['Salary'])

train_x.head()
train=[]
gc.collect()

test_x.head()
test_y[:10]

train_x.head()
train_y[:10]



#Born - datetime needs to be changed to days since a set date
#days form birth to season start

def elapsed_days(start, end=datetime(2016,10,12)):
	""" calcualte the number of days start and end dates"""
	x = (end - start)
	return x.days

#
train_x['age_season_start'] = train_x.apply(lambda x: 
	elapsed_days(datetime.strptime(x['Born'], '%y-%m-%d')) ,axis=1)

test_x['age_season_start'] = test_x.apply(lambda x: 
	elapsed_days(datetime.strptime(x['Born'], '%y-%m-%d')) ,axis=1)



# Drop the city, province and Cntry cols, will include nationality but all these
# seemed redundant on the initial rf and XGBoost models

drop_cols = ['City', 'Pr/St', 'Cntry', 'Last Name', 'First Name', 'Team', 'Born']

test_x.drop(drop_cols, axis = 1, inplace = True)

train_x.drop(drop_cols, axis = 1, inplace = True)



#check the data types of the remaining columns
train_x.dtypes
for i in train_x.dtypes:
	print(i)


#Categoricals:
cat_attribs = ['Nat', 'Hand', 'Position']

num_attribs = list(train_x.drop(cat_attribs,axis=1).columns)


class DataFrameSelector(BaseEstimator, TransformerMixin):
	""" this class will select a subset of columns,
		pass in the numerical or categorical columns as 
		attribute names to get just those columns for processing"""
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		return X[self.attribute_names]



#build my own to binarize multiple labels at once, 
#then implement it in the cat_pipeline
"""
x = MultiLabelBinarizer(train_x[cat_attribs])

tempdf = pd.get_dummies(train_x, columns=cat_attribs)

encoder = LabelBinarizer()
x =encoder.fit_transform(train_x['Nat'])


class_test = MultiColBinarize()
class_test.fit_transform(train_x[cat_attribs])

class_test.transform()
"""


class MultiColBinarize(BaseEstimator, TransformerMixin):
	""" take a df with multiple categoricals
		one hot encode them all and return the numpy array"""
	def __init__(self, alter_df= True):
		self.alter_df = alter_df
	def fit(self, X, y=None):
		"""load the data in, initiate the binarizer for each column"""
		self.X = X
		self.cols_list = list(self.X.columns)
		self.binarizers = []
		for i in self.cols_list:
			encoder = LabelBinarizer()
			encoder.fit(self.X[i])
			self.binarizers.append(encoder)
		return self
	def transform(self, X):
		""" for each of the columns, use the existing binarizer to make new cols """		
		self.X = X
		self.binarized_cols = self.binarizers[0].transform(self.X[self.cols_list[0]])
		self.classes_ = list(self.binarizers[0].classes_)
		for i in range(1,len(self.cols_list)):
			binarized_col = self.binarizers[i].transform(self.X[self.cols_list[i]])
			self.binarized_cols = np.concatenate((self.binarized_cols , binarized_col), axis = 1)
			self.classes_.extend(list(self.binarizers[i].classes_))
		return self.binarized_cols



num_pipeline = Pipeline([
		('selector', DataFrameSelector(num_attribs)),
		('imputer', Imputer(strategy="median")),
		('std_scaler', StandardScaler()),
	])

# select the categorical columns, binarize them 
cat_pipeline = Pipeline([
		('selector', DataFrameSelector(cat_attribs)),
		('label_binarizer', MultiColBinarize()),
	])



#####
# impute missing values and prepare the categoricals for ml algorithms
#####


train_num_processed = num_pipeline.fit_transform(train_x)
train_cat_processed = cat_pipeline.fit_transform(train_x)

train_x_clean =  np.concatenate((train_num_processed,train_cat_processed),axis=1)


#need to just transform the test, we impute based on the training data!

test_num_processed = num_pipeline.transform(test_x)
test_cat_processed = cat_pipeline.transform(test_x)

test_x_clean =  np.concatenate((test_num_processed,test_cat_processed),axis=1)


#check that the number of columns are the same for both
train_x_clean.shape
test_x_clean.shape


""" imputation is successfully completed, on to the modelling """

##########
# support vector machine
##########


svm_reg = SVR(kernel="linear")


svr_param_grid = [
		{'kernel': ['rbf','linear'], 'C': [1.0, 10., 100., 1000.0],
		'gamma': [0.01, 0.1,1.0]}
	]


svm_grid_search = GridSearchCV(svm_reg, svr_param_grid, cv=5,
						scoring='neg_mean_squared_error')

svm_grid_search.fit(train_x_clean, train_y)

svm_grid_search.best_params_

svm_grid_search.best_estimator_

cvres = svm_grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(np.sqrt(-mean_score), params)



##########
# Random forest regression
##########


forest_reg = RandomForestRegressor(random_state=42)

rf_param_grid = [
	{'n_estimators': [3, 10, 30,100,300,1000], 'max_features': [2, 4, 6, 8]},
	{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
rf_grid_search = GridSearchCV(forest_reg, rf_param_grid, cv=5,
						   scoring='neg_mean_squared_error')
rf_grid_search.fit(train_x_clean, train_y)

rf_grid_search.best_params_

rf_grid_search.best_estimator_

cvres = rf_grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(np.sqrt(-mean_score), params)


##########
# XGBoost model
##########


XGBoost_reg = xgb.XGBRegressor()

#note all the params below must be wrapped in lists
xgb_param_grid  = [{'min_child_weight': [20,25,30], 
					'learning_rate': [0.1, 0.2, 0.3], 
					'colsample_bytree': [0.9], 
					'max_depth': [5,6,7,8], 
					'reg_lambda': [1.], 
					'nthread': [-1], 
					'n_estimators': [100,1000,2000],
					'early_stopping_rounds':50,
					'objective': ['reg:linear']}]


xgb_grid_search = GridSearchCV(XGBoost_reg, xgb_param_grid, cv=5,
					scoring='neg_mean_squared_error', n_jobs=1)

xgb_grid_search.fit(train_x_clean, train_y)


xgb_grid_search.best_params_

xgb_grid_search.best_estimator_

cvres = xgb_grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(np.sqrt(-mean_score), params)




# test the above 3 models, retrain on the top set of paramaters


#SVM
opt_svm_params = {'C': 1000.0, 
				'gamma': 0.01, 
				'kernel': 'linear'}

#need the ** to unpack the dictonary so all the params don't get assigned to one
opt_svm_reg = SVR(**opt_svm_params)

opt_svm_reg.fit(train_x_clean, train_y)


#RF
opt_rf_params= {'max_features': 8, 'n_estimators': 100}

opt_forest_reg = RandomForestRegressor(**opt_rf_params, random_state=42)

opt_forest_reg.fit(train_x_clean, train_y)


#XGB
opt_xgb_params = {'colsample_bytree': 0.9,
				'learning_rate': 0.1,
				'max_depth': 7,
				'min_child_weight': 30,
				'n_estimators': 1000,
				'nthread': -1,
				'objective': 'reg:linear',
				'reg_lambda': 1.0}


opt_XGBoost_reg = xgb.XGBRegressor(**opt_xgb_params)

opt_XGBoost_reg.fit(train_x_clean, train_y)



y1 = opt_XGBoost_reg.predict(test_x_clean)
y2 = opt_svm_reg.predict(test_x_clean)
y3 = opt_forest_reg.predict(test_x_clean)


"""
do this for each:

median_mse= mean_squared_error(test_y,meadian_guess)

median_rmse = np.sqrt(median_mse)
median_rmse
"""

# then find a way to optimize their combination into a single model
#combine the three optimal predictors into a single sklearn class that spits
#out predicted values, use this with a tuning param that changes the weights of
#the models and use cross validation function to get the scores.


class ensemble_predictor(BaseEstimator, TransformerMixin):
	""" take in a dataset and train it with three models,
		combining the outputs to make predictions"""
	def __init__(self, weights= { 'xgb': 0.33, 'rf': 0.33, 'svm' : 0.34}):
		self.weights = weights
		self.opt_xgb_params = {'colsample_bytree': 0.9,
					'learning_rate': 0.1,
					'max_depth': 7,
					'min_child_weight': 30,
					'nthread': -1,
					'objective': 'reg:linear',
					'reg_lambda': 1.0}
		self.opt_svm_params = {'C': 1000.0, 
				'gamma': 0.01, 
				'kernel': 'linear'}
		self.opt_rf_params= {'max_features': 8, 'n_estimators': 100}

	def fit(self, X, y):
		"""load the data in, initiate the models"""
		self.X = X
		self.y = y
		self.opt_XGBoost_reg = xgb.XGBRegressor(**self.opt_xgb_params)
		self.opt_forest_reg = RandomForestRegressor(**self.opt_rf_params)
		self.opt_svm_reg = SVR(**self.opt_svm_params)
		""" fit the models """
		self.opt_XGBoost_reg.fit(self.X ,self.y)
		self.opt_forest_reg.fit(self.X ,self.y)
		self.opt_svm_reg.fit(self.X ,self.y)
	def predict(self, X2):
		""" make the predictions for the models, combine based on weights """
		self.y_xgb = self.opt_XGBoost_reg.predict(X2)
		self.y_rf = self.opt_forest_reg.predict(X2)
		self.y_svm = self.opt_svm_reg.predict(X2)
		""" multiply the predictions by their weights, return optimal """
		self.prediction = self.y_xgb * self.weights['xgb'] \
						+ self.y_rf * self.weights['rf'] \
						+ self.y_svm * self.weights['svm']
		return self.prediction

weight_variants = [
{ 'xgb': 0.33, 'rf': 0.33, 'svm' : 0.34},
{ 'xgb': 0.9, 'rf': 0.05, 'svm' : 0.05},
{ 'xgb': 0.8, 'rf': 0.1, 'svm' : 0.1},
{ 'xgb': 0.5, 'rf': 0.3, 'svm' : 0.2},
{ 'xgb': 0.3, 'rf': 0.2, 'svm' : 0.5},
{ 'xgb': 0.3, 'rf': 0.5, 'svm' : 0.2}
]



#determine the optimal weights for the different models via cross validation
for params in weight_variants:
	model = ensemble_predictor(weights = params)
	ensemble_score = cross_val_score(model, train_x_clean, train_y,
							scoring="neg_mean_squared_error", cv=5)
	ensemble_rmse = np.sqrt(-ensemble_score)
	print('%s\t %s'% (params, ensemble_rmse.mean()))

#winner
# {'xgb': 0.8, 'rf': 0.1, 'svm': 0.1}	 1322950.1668

#try again with the new weight variants, tuned in towards the optimal numbers
weight_variants = [
{ 'xgb': 0.8, 'rf': 0.15, 'svm' : 0.05},
{ 'xgb': 0.8, 'rf': 0.05, 'svm' : 0.15},
{ 'xgb': 0.82, 'rf': 0.09, 'svm' : 0.09},
{ 'xgb': 0.79, 'rf': 0.105, 'svm' : 0.105},
{ 'xgb': 0.79, 'rf': 0.11, 'svm' : 0.1},
{ 'xgb': 0.79, 'rf': 0.1, 'svm' : 0.11}
]


#{'xgb': 0.8, 'rf': 0.15, 'svm': 0.05}	 1322424.6932
#
weights = {'xgb': 0.8, 'rf': 0.15, 'svm': 0.05}

opt_model = ensemble_predictor(weights)
opt_model.fit(train_x_clean, train_y)
final_predictions = opt_model.predict(test_x_clean)


opt_mean_squared_error = mean_squared_error(test_y,final_predictions)

opt_rmse = np.sqrt(opt_mean_squared_error)
opt_rmse

#1,546,809

meadian_guess = [np.median(test_y) for x in test_y]
#925000

median_mse= mean_squared_error(test_y,meadian_guess)

median_rmse = np.sqrt(median_mse)
median_rmse
#2878624
"""
#therefore our model is about 1.3 million dollars closer on average than 
#guessing by just the median alone

#the cross validation was off by only 472073, which is suggestive of over fit as this is
#under a third of our final rmse on the test data.

this mixed model was better then previous iterations, when we ran
Random Forest regression the model was off by an average of $1,578,497
and with XGBoost alone the model was only slightly improved at $1,574,073
When we combined models here, we see that we are about $25,000 closer on average,
which is a slight improvement, but an improvement nonetheless!

"""

