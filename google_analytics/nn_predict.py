import pandas as pd
import numpy as np
import tensorflow as tf

#load the pickled matricies
X_train = np.load('X_train.dat')
y_train = np.load('y_train.dat')
X_test = np.load('X_test.dat')


#need to change all the columns from float64 to float32
X_train = np.float32(X_train)
y_train = np.float32(y_train)
X_test = np.float32(X_test)


config = tf.contrib.learn.RunConfig(tf_random_seed=42)

feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)

dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[150,300,900,300,150], n_classes=2,
                                         feature_columns=feature_cols, config=config)

dnn_clf.fit(X_train, y_train, batch_size=50, steps=40000)


dnn_y_pred = dnn_clf.predict(X_test)

test_y = list(dnn_y_pred)


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
submission.to_csv('cam_dnn_pred2.csv', index = False)

"""
I suspect this may not work the first time around.... can try to downsample the 0s
in order to give a more balanced dataset.... possibly 3:1 0s:purchases
"""
