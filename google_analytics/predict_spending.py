import pandas as pd
import numpy as np


all_train = pd.read_csv('./data/train_cleaned.csv')

all_train.head() #there was a json one I missed, double back time :(

#need to go through and clean the columns
all_train.describe()

all_train.columns

#51 columns

all_train['adwordsClickInfo'][0]


all_train['transactionRevenue']


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

for i in list(dups):
	ex_dup = all_train[all_train['fullVisitorId'] == i]

	for col in ex_dup.columns:
		if len(ex_dup[col].unique()) != 1 and col not in flex_cols:
			flex_cols.append(col)

flex_cols