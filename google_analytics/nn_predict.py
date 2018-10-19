import pandas as pd
import numpy as np

#load the pickled matricies
X_train = np.load('X_train.dat')
y_train = np.load('y_train.dat')
X_test = np.load('X_test.dat')


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

final_test['fullVisitorId'] = final_test['fullVisitorId'].astype('str')

#group by id
final_pred = final_test[['fullVisitorId']].copy()
final_pred['test_pred'] = test_y
final_by_ind =  final_pred.groupby(['fullVisitorId']).sum()


#move index to a col
final_by_ind = final_by_ind.reset_index()

#merge the predictions with the sample sub
submission = pd.read_csv('./data/sample_submission.csv')
submission = submission.merge(final_by_ind, on = 'fullVisitorId', how = 'left')

#fill nas and move to right column name
submission['PredictedLogRevenue'] = submission['test_pred'].fillna(0.0)
submission = submission.drop(['test_pred'], axis = 1)

#submit the output
submission.to_csv('cam_lightgbm_pred2.csv', index = False)


