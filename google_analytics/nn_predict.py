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


"""
I suspect this may not work the first time around.... can try to downsample the 0s
in order to give a more balanced dataset.... possibly 3:1 0s:purchases
or:upsample the 1+s, this is likely okay b/c the training algo works in batches
"""

nz = y_train != 0

nz_index = []
for i, x in enumerate(nz):
	if x == True:
		nz_index.append(i)


len(nz_index)/len(y_train)
#1.2% of data non zero

X_train_non_zeros = X_train[nz_index]

y_train_non_zeros = y_train[nz_index]


upsample_X_train = np.repeat(X_train_non_zeros, 99 , axis = 0)
upsample_X_train.shape

upsample_y_train = np.repeat(y_train_non_zeros, 99 , axis = 0)
upsample_y_train.shape


X_train_changed = np.concatenate((X_train, upsample_X_train), axis = 0)
X_train_changed.shape[0] == X_train.shape[0] + upsample_X_train.shape[0]

y_train_changed = np.append(y_train,upsample_y_train)
y_train_changed.shape


####
# dnn
####

feature_cols = [tf.feature_column.numeric_column("X", shape=[X_train_changed.shape[1]])]

#hidden_units=[150,300,600,300,150]
dnn_clf = tf.estimator.DNNRegressor(hidden_units=[150,300,600,300,150],
										feature_columns=feature_cols,
										optimizer=tf.train.ProximalAdagradOptimizer(
											learning_rate=0.001,
											l1_regularization_strength=0.001
											))

input_fn = tf.estimator.inputs.numpy_input_fn(
	 					x={"X": X_train_changed}, 
	 					y=y_train_changed, 
	 					num_epochs=20000, batch_size=100, shuffle=True)

dnn_clf.train(input_fn=input_fn)

#the output is a generator object
dnn_y_pred = dnn_clf.predict(X_test)

test_y = list(dnn_y_pred)


# sum the predictions using the defined formula to get a revenue by user metric
# aggregate on 'fullVisitorId' 
# final_test['fullVisitorId' ]


final_test = pd.read_csv('./data/test_cleaned.csv')
final_test['fullVisitorId'] = final_test['fullVisitorId'].astype('str')


final_pred = final_test[['fullVisitorId']].copy()
final_pred['test_pred'] = test_y


#group by id
final_by_ind =  final_pred.groupby(['fullVisitorId']).sum()
#move index to a col
final_by_ind = final_by_ind.reset_index()

#merge the predictions with the sample sub
submission = pd.read_csv('./data/sample_submission.csv')
submission = submission.merge(final_by_ind, on = 'fullVisitorId', how = 'left')
#fill nas and move to right column name
submission['PredictedLogRevenue'] = submission['test_pred'].fillna(0.0)
submission = submission.drop(['test_pred'], axis = 1)


def set_min_zero(x):
	if x < 0:
		return 0
	else:
		return x

submission['PredictedLogRevenue'] = submission['PredictedLogRevenue'].apply(
									lambda x: set_min_zero(x))


#submit the output
submission.to_csv('cam_dnn_upsample1_floor.csv', index = False)
