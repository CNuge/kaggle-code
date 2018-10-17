import pandas as pd
import numpy as np


#read in the sample submission
submission = pd.read_csv('./data/sample_submission.csv')

#read in the three matricies and extract just the np arrays
X_train=pd.read_csv('X_train.csv', sep=',', header=None)
X_train = X_train.values
y_train=pd.read_csv('y_train.csv', sep=',', header=None)
y_train = y_train.values
X_test=pd.read_csv('X_test.csv', sep=',', header=None)
X_test = X_test.values




#####
#
# put the model training here
#
#####


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