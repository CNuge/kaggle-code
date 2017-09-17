""" these are generic functions for specific munging tasks and data assessment"""
""" see the pipeline below for standard treatment of data using scikit learn builtins """

def display_scores(scores):
    """ pass in a list of validation output scores and 
        this returns the data in formatted manner """
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('std_dev:', scores.std())
    

def fill_median(dataframe, cols):
	"""impute the mean for a list of columns in the dataframe"""
	for i in cols:
		dataframe[i].fillna(dataframe[i].median(skipna=True), inplace = True)
	return dataframe

def cols_with_missing_values(dataframe):
	""" query a dataframe and find the columns that have missing values"""
	return list(dataframe.columns[dataframe.isnull().any()])

def fill_value(dataframe, col, val):
	"""impute the value for a list column in the dataframe"""
	""" use this to impute the median of the train into the test"""
	dataframe[i].fillna(val, inplace = True)
	return dataframe




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





#############
# Generic pipeline for imputation of numericals(median) and one-hot coding of categoricals
# use below as a starting point for the processing of data before ml use
#can tailor with some feature engineering in the combined attributes adder function
#############

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Imputer
from sklearn.pipeline import FeatureUnion


# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
	""" this class will select a subset of columns,
		pass in the numerical or categorical columns as 
		attribute names to get just those columns for processing"""
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		return X[self.attribute_names].values



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
	""" this is a custom class to make alterations to the numeric variables
		in a dataframe in preparation for machine learning algorithm use """
	def __init__(self, alter_df= True): # no *args or **kargs
		self.alter_df = alter_df
	def fit(self, X, y=None):
		return self  # nothing else to do
	def transform(self, X, y=None):
		if self.alter_df:
			""" code in aterations to df here """
			return np.c_[X,  #THINGS you've added
						]		
		else:
			""" if alter_df=False, then just return the df """
			return np.c_[X]






#list the numeric and then list the categoricals
cat_attribs = ["ocean_proximity"]
num_attribs = list(X_train.drop(cat_attribs,axis=1).columns)


#below is a pipeline for  numerical values, it will impute median and standardize scale
#the commented line is an optional one, to add attributes based on combining columns
# see above and alter CombinedAttributesAdder() for the given task.
num_pipeline = Pipeline([
		('selector', DataFrameSelector(num_attribs)),
		('imputer', Imputer(strategy="median")),
		#('attribs_adder', CombinedAttributesAdder()),
		('std_scaler', StandardScaler()),
	])

#this is the pipline for numerical variables, it selects the categorical columns
cat_pipeline = Pipeline([
		('selector', DataFrameSelector(cat_attribs)),
		('label_binarizer', LabelBinarizer()),
	])

#this calls the two pipelines and merges the outputs into one
full_pipeline = FeatureUnion(transformer_list=[
		("num_pipeline", num_pipeline),
		("cat_pipeline", cat_pipeline),
	])

#######
#usage
#######

#X_train_clean = full_pipeline.fit_transform(X_train)
#X_test_clean = full_pipeline.transform(X_train)

#check that the number of columns are the same for both
#X_train_clean.shape
#X_test_clean.shape





#####
# make combine predictions based on several ml functions
#below example is set up for rf, xgb and svm combined model
#####



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
		self.y_xgb = opt_XGBoost_reg.predict(X2)
		self.y_rf = opt_forest_reg.predict(X2)
		self.y_svm =opt_svm_reg.predict(X2)
		""" multiply the predictions by their weights, return optimal """
		self.prediction = self.y_xgb * self.weights['xgb'] \
						+ self.y_rf * self.weights['rf'] \
						+ self.y_svm * self.weights['svm']
		return self.prediction

#pass in a list of weight paramater options
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
