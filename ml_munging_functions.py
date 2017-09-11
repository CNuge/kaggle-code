
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


#############
# Generic pipeline for imputation of numericals and one hot coding of categoricals
#use the below as a starting point for the processing of data before ml use
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
			return np.c_[X, rooms_per_household, population_per_household,
						 bedrooms_per_room]
		else:
			""" if alter_df=False, then just return the df """
			return np.c_[X, rooms_per_household, population_per_household]




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
#X_test_clean = full_pipeline.fit_transform(X_train)

#check that the number of columns are the same for both
#X_train_clean.shape
#X_test_clean.shape

