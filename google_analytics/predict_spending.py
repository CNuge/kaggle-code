import pandas as pd
import numpy as np


import gc
from sklearn.preprocessing import LabelBinarizer


####
# load the data
####

all_train = pd.read_csv('./data/train_cleaned.csv')
#all_train.head() 

final_test = pd.read_csv('./data/test_cleaned.csv')
#final_test.head()

submission = pd.read_csv('./data/sample_submission.csv')
#submission.head()

####
# check submission length
####

#it is lower than the number of ids in the test set?
len(submission['fullVisitorId']) == len(set(submission['fullVisitorId']))
len(set(submission['fullVisitorId'])) == len(set(final_test['fullVisitorId']))


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
# drop flat cols for both the train and test data
####

#should drop the flatline columns from the df
all_train = all_train.drop(flatline, axis = 1)
all_train.shape

final_test = final_test.drop(flatline, axis=1)
final_test.shape

for i in list(all_train.columns):
	if i not in list(final_test.columns):
		print(i)

all_train = all_train.drop('campaignCode', axis=1)


#drop campaign code, transactionRevenue is what we are trying to predict

#######
# finish cleaning the columns
#######
all_train.head()
final_test.head()


####
# numeric
####


'fullVisitorId' #removed form numeric, this is just the id
'transactionRevenue' #this is the response variable we want to predict

numeric = [ 'newVisits',
			 'pageviews',
			 'transactionRevenue',
			 ]

def fill_and_adj_numeric(df):
	#there are NA for page views, fill median for this == 1
	df.isTrueDirect.fillna(df.pageviews.median(), inplace = True)

	#are boolean, fill NaN with zeros, add to categorical
	df.isTrueDirect.fillna(0, inplace = True)
	df.bounces.fillna(0, inplace = True)
	df.newVisits.fillna(0, inplace = True)

	for col in ['isTrueDirect', 'bounces', 'newVisits']:
		df[col] = df[col].astype(int)

	return df

all_train = fill_and_adj_numeric(all_train)
final_test = fill_and_adj_numeric(final_test)

####
# other
####
other = ['Unnamed: 0',
			 'date',
			 ,
			 'referralPath']

drop_other = ['visitId',
				'Unnamed: 0',
				'campaignCode'
				'referralPath']



numeric_other = ['visitNumber', 
					'hits',
					'visits']

categorical_other = ['isMobile',
						'adContent',
						]

all_train['date'] #this needs to be processed with datetime
all_train['visitStartTime'] 





#######

####
# categorical
####

categorical = 	['channelGrouping',
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
				 'adwordsClickInfo',
				 'campaign',
				 'keyword',
				 'medium',
				 'source']

all_train.adwordsClickInfo #this one isn't fixed!
final_test.adwordsClickInfo

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
	if len(all_train[col].unique()) > 1:
		bin_col_all_train, bin_col_final_test = binarize_col(all_train, final_test, col)

		if len(train_bins) == 0:
			train_bins = bin_col_all_train	
			test_bins =	bin_col_final_test
		else:
			train_bins = np.c_[train_bins, bin_col_all_train]
			test_bins = np.c_[test_bins, bin_col_final_test]
	gc.collect()

train_bins.shape
test_bins.shape


#drop the non binary categories and the 

all_train = all_train.drop(categorical, axis = 1)


final_test = final_test.drop(final_test, axis = 1)


# isolate the response variable
y_train = all_train['transactionRevenue']
y_test = final_test['transactionRevenue']

X_train = all_train.drop(['fullVisitorId','transactionRevenue'],axis = 1).values
X_train = np.c_[X_train, train_bins]

X_test = all_train.drop(['fullVisitorId','transactionRevenue'],axis = 1).values
X_test = np.c_[X_test, test_bins]




####
# TODO
####
#1. turn datetimes into numerics from a start date
#2. fill na on all the categoricals
#3. drop all the other columns not to be used
#4. treat numeric others with numerics
#5. get the label binarizer to not crash







######
# build the first model
######

#split off 20% of the train data for evaluation - validation

#use a k fold cross validation approach on the remaining all_trian dataset
	#set up an xgboost model for the dataset via cv
	#see how well the model works on the 20% validation data

#run a grid search to pick the best params


#traing the model on the full all_train dataset


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

#after that -> try some mode models and ensemble the solutions.




