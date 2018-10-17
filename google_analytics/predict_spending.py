import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import xgboost as xgb

import gc
import time


####
# load the data
####

all_train = pd.read_csv('./data/train_cleaned.csv')
#all_train.head() 

final_test = pd.read_csv('./data/test_cleaned.csv')
#final_test.head()

submission = pd.read_csv('./data/sample_submission.csv')
#submission.head()

"""
####
# check submission length
####

#it is lower than the number of ids in the test set?
len(submission['fullVisitorId']) == len(set(submission['fullVisitorId']))
len(set(submission['fullVisitorId'])) == len(set(final_test['fullVisitorId']))
"""

"""
####
# explore what we are looking at
####

#need to go through and clean the columns
all_train.describe()

all_train.columns
#51 columns
all_train['adwordsClickInfo'][0] #this is still json buy okay
type(all_train['transactionRevenue'][0])  == np.float64#this is the one we are trying to predict
all_train.columns
"""

####
# scan columns and classify
####

numeric = []
categorical = []
flatline = []
other = []

for col in all_train.columns:
	if type(all_train[col][0]) == str:
		#categorical
		if len(all_train[col].unique()) > 1:
			categorical.append(col)
		else:
			flatline.append(col)
	elif type(all_train[col][0]) == int or type(all_train[col][0]) == np.float64:
		#numeric
		numeric.append(col)
	else:
		other.append(col)

numeric
categorical
flatline
other

####
# other columns
####
drop_other = ['visitId',
				'Unnamed: 0',
				'campaignCode',
				'referralPath',
				'adwordsClickInfo',
				'adContent']


numeric_other = ['visitNumber', 
					'hits',
					'visits']

categorical_other = ['isMobile',]



####
# drop flat cols for both the train and test data
####

flatline.extend(drop_other)
#should drop the flatline columns from the df
all_train = all_train.drop(flatline, axis = 1)
all_train.shape

flatline = [x for x in flatline if x != 'campaignCode' ]
final_test = final_test.drop(flatline, axis=1)
final_test.shape

for i in list(all_train.columns):
	if i not in list(final_test.columns):
		print(i)


#######
# finish cleaning the columns
#######
all_train.head()
final_test.head()


####
# numeric
####


#'fullVisitorId' #removed form numeric, this is just the id
#'transactionRevenue' #this is the response variable we want to predict

numeric = [ 'newVisits',
			 'pageviews',
			 'transactionRevenue',
			 ]

numeric.extend(numeric_other)

all_train['transactionRevenue'].fillna(0, inplace = True)

def fill_and_adj_numeric(df):
	#there are NA for page views, fill median for this == 1
	df.pageviews.fillna(df.pageviews.median(), inplace = True)

	df.hits.fillna(df.hits.median(), inplace = True)
	df.visits.fillna(df.visits.median(), inplace = True)

	#are boolean, fill NaN with zeros, add to categorical
	df.isTrueDirect.fillna(0, inplace = True)
	df.bounces.fillna(0, inplace = True)
	df.newVisits.fillna(0, inplace = True)
	df.visitNumber.fillna(1, inplace = True)

	for col in ['isTrueDirect', 'bounces', 'newVisits']:
		df[col] = df[col].astype(int)

	return df

all_train = fill_and_adj_numeric(all_train)
final_test = fill_and_adj_numeric(final_test)


####
# datetime columns
##

all_train['date'] #this needs to be processed with datetime

def parseDateCol(df, date_col):
	""" takes the date column and adds new columns with the features:
		yr, mon, day, day of week, day of year """
	df['datetime'] = df.apply(lambda x : time.strptime(str(x[date_col]),  "%Y%M%d"), axis = 1)
	print('parsing year')
	df['year'] = df.apply(lambda x : x['datetime'].tm_year, axis = 1)
	print('parsing month')
	df['month'] = df.apply(lambda x :x['datetime'].tm_mon , axis = 1)
	print('parsing days (*3 versions)')
	df['mday'] = df.apply(lambda x : x['datetime'].tm_mday, axis = 1)
	df['wday'] = df.apply(lambda x : x['datetime'].tm_wday , axis = 1)
	df['yday'] = df.apply(lambda x : x['datetime'].tm_yday , axis = 1)

	#drop date and datetime
	df = df.drop([date_col, 'datetime'], axis = 1)
	
	return df

all_train = parseDateCol(all_train, 'date')

final_test = parseDateCol(final_test, 'date')



####
# categorical
####

categorical = ['channelGrouping',
				 'sessionId',
				 'browser',
				 'deviceCategory',
				 'operatingSystem',
				 'city',
				 'continent',
				 'country',
				 'metro',
				 'networkDomain',
				 'region',
				 'subContinent',
				 'campaign',
				 'keyword',
				 'medium',
				 'source']

categorical.extend(categorical_other)


with_na = []
for col in categorical:
	if all_train[col].isnull().any() :
		with_na.append(col)		


####
# fill na for all the categoricals with the 'None' if string or mode if bool
####

#most common value to fill the na
all_train.keyword.fillna('(not provided)', inplace = True)

def binarize_col(train, test, col):
	encoder = LabelBinarizer()

	cat_train_1hot = encoder.fit_transform(train[col])
	
	cat_test_1hot = encoder.transform(test[col])

	return cat_train_1hot, cat_test_1hot


train_bins = []
test_bins = []
#this is crashing... need a little more memory I think
for col in categorical:
	if len(all_train[col].unique()) > 1 and len(all_train[col].unique()) < 500:

		print(f'binarizing:{col}\tunique: {len(all_train[col].unique())}')

		bin_col_all_train, bin_col_final_test = binarize_col(all_train, final_test, col)

		if len(train_bins) == 0:
			print('initializing np matrix')
			train_bins = bin_col_all_train	
			test_bins =	bin_col_final_test
		else:
			print('appending to np matrix')
			train_bins = np.c_[train_bins, bin_col_all_train]
			test_bins = np.c_[test_bins, bin_col_final_test]
	gc.collect()

train_bins.shape
test_bins.shape


#drop the non binarized categorical columns and the housekeeping ones ones to the 
#the train and test sets for sklearn

all_train = all_train.drop(categorical, axis = 1)
final_test = final_test.drop(categorical, axis = 1)


# isolate the response variable
y_train = all_train['transactionRevenue'].values


all_train.columns

for x in all_train.columns:
	print(x, all_train[x].dtype)


X_train = all_train.drop(['fullVisitorId','transactionRevenue'],axis = 1).values
X_train = np.c_[X_train, train_bins]
X_train.shape


X_test = final_test.drop(['fullVisitorId'],axis = 1).values
X_test = np.c_[X_test, test_bins]
X_test.shape


np.savetxt("X_train.csv", X_train, delimiter=",")
np.savetxt("y_train.csv", y_train, delimiter=",")
np.savetxt("X_test.csv", X_test, delimiter=",")

######
# build the first model
######

#split off 20% of the train data for evaluation - validation

#use a k fold cross validation approach on the remaining all_trian dataset
	#set up an xgboost model for the dataset via cv
	#see how well the model works on the 20% validation data

#run a cv search to pick the num of rounds


dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test)

y_mean = np.mean(y_train) #this is the baseline prediction, the mean
y_median = np.median(y_train) #this is the baseline prediction, the mean

#possibly switch this to the median

xgb_params = {'eta' :  0.05,
                'max_depth' :  8,
                'subsample' : 0.80, 
                'objective' :  'reg:linear',
                'eval_metric' : 'rmse',
                'base_score' :  y_median,}



#before cv, can try with just a super simple train to make sure it is working
test_model = xgb.train(xgb_params, dtrain, 
                  num_boost_round = 2,
                  verbose_eval=1)


cv_result = xgb.cv(xgb_params, dtrain, 
					nfold=5, 
					num_boost_round=10, 
					#early_stopping_rounds=50, 
					verbose_eval=1, show_stdv=False)

#if works, swtict to

cv_result = xgb.cv(xgb_params, dtrain, 
					nfold=5, 
					num_boost_round=2000, 
					early_stopping_rounds=50, 
					verbose_eval=0, show_stdv=False)


num_boost_rounds = len(cv_result)

#traing the model on the full all_train dataset
model = xgb.train(xgb_params, dtrain, 
                  num_boost_round = num_boost_rounds)



#make predictions on the test data

test_y = model.predict(X_test)

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
submission


#after that:
	#
	#try some mode models and ensemble the solutions.


#improve:
# after original model submission is validated, then take the model and 
# run girdsearch cv to try a bunch of different hyperparams
from sklearn.model_selection import GridSearchCV

param_test1 = {
 'max_depth':range(3,12,2),
 'min_child_weight':range(1,6,2)
}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
				 		min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 						objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 						param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch1.fit(dtrain)

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


