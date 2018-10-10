import pandas as pd
import numpy as np

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

"""

####
# handling duplicated ids
####

len(all_train['fullVisitorId'])
len(all_train['fullVisitorId'].unique())
len(all_train['fullVisitorId'][all_train['fullVisitorId'].duplicated()])


#several instances where the id is not unique, these are repeat visitors
dups = all_train['fullVisitorId'][all_train['fullVisitorId'].duplicated()]

test_id = list(dups)[0]


#find which columns in the data are varying between the duplicate visitor ids
#these data must somehow be collapsed into a single entry... will be case by 
#case depending on numeric/ categorical/ NaN situation.
flex_cols = []

for i in list(dups)[:100]:
	ex_dup = all_train[all_train['fullVisitorId'] == i]

	for col in ex_dup.columns:
		if len(ex_dup[col].unique()) != 1 :
			flex_cols.append(col)

flex_cols = set(flex_cols)

len(flex_cols) #thirty of them are varying across the two visits


#numeric flex cols:
#see which make sense to take average of and which should be summers

#categorical flex cols:
#see which should be merged somehow (possibly into a numeric count)
#and for which a single value should be kept


#split the date columns into hour/day etc.

"""

#idea: don't merge across users - just leave them separate as individual sessions
#and train the algorithm on that data. Then for the test data make the recommendations
#at a session level and then sum the results to make the final prediction.

#######
# finish cleaning the columns
#######
all_train.head()
final_test.head()

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

#most common value to fill the na
all_train.keyword.fillna('(not provided)', inplace = True)


def binarize_col(train, test, col):
	encoder = LabelBinarizer()

	cat_train_1hot = encoder.fit_transform(train[col])
	
	cat_test_1hot = encoder.transform(test[col])

	return cat_train_1hot, cat_test_1hot


train_bins = []
test_bins = []
for col in categorical:
	if len(all_train[col].unique()) > 1:
		bin_col_all_train, bin_col_final_test = binarize_col(all_train, final_test, col)

		if len(train_bins) == 0:
			train_bins = bin_col_all_train	
			test_bins =	bin_col_final_test
		else:
			train_bins = np.c_[train_bins, bin_col_all_train]
			test_bins = np.c_[test_bins, bin_col_final_test]

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

# sum the predictions using the defined formula to get a revenue by user metric
# aggregate on 'fullVisitorId' 
# final_test['fullVisitorId' ]


#submit


#after that -> try some mode models and ensemble the solutions.



