import pandas as pd
import numpy as np

####
# load the data
####

all_train = pd.read_csv('./data/train_cleaned.csv')

all_train.head() 

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

#should drop the flatline columns from the df
all_train = all_train.drop(flatline, axis = 1)

all_train.shape

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