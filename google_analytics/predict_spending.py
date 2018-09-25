import pandas as pd
import numpy as np

####
# load the data
####

all_train = pd.read_csv('./data/train_cleaned.csv')
all_train.head() 

final_test = pd.read_csv('./data/test_cleaned.csv')
final_test.head()

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
