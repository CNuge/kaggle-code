

""" this script makes classifications based on a deep neural network 
	network is trained with tensorflow, and coded using the sklearn wrapper"""



import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
#from tensorflow.contrib.learn.python import SKCompat

# Load the data

test_dat = pd.read_csv('test.csv')
train_dat = pd.read_csv('train.csv')
submission = pd.read_csv('sample_submission.csv')


train_y = train_dat['target']
train_x = train_dat.drop(['target', 'id'], axis = 1)
test_dat = test_dat.drop(['id'], axis = 1)

merged_dat = pd.concat([train_x, test_dat],axis=0)

for c, dtype in zip(merged_dat.columns, merged_dat.dtypes): 
	if dtype == np.float64:     
		merged_dat[c] = merged_dat[c].astype(np.float32)

cat_features = [col for col in merged_dat.columns if col.endswith('cat')]
for column in cat_features:
	temp=pd.get_dummies(pd.Series(merged_dat[column]))
	merged_dat=pd.concat([merged_dat,temp],axis=1)
	merged_dat=merged_dat.drop([column],axis=1)


numeric_features = [col for col in merged_dat.columns if '_calc_' in  str(col)]
numeric_features = [col for col in numeric_features if '_bin' not in str(col)]

scaler = StandardScaler()
scaled_numerics = scaler.fit_transform(merged_dat[numeric_features])
scaled_num_df = pd.DataFrame(scaled_numerics, columns =numeric_features )


merged_dat = merged_dat.drop(numeric_features, axis=1)



merged_dat = np.concatenate((merged_dat.values,scaled_num_df), axis = 1)


train_x = merged_dat[:train_x.shape[0]]
test_dat = merged_dat[train_x.shape[0]:]



"""
train the neural network
"""


config = tf.contrib.learn.RunConfig() # not shown in the config

#create a set of real valued columns from training set
feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(train_x)

#create the DNN classifier. DNN == deep neural network
#understand these params and alter accordingly
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[150,150,150,100,50], n_classes=2,
                                         feature_columns=feature_cols, config=config)

#hidden units means the number of units per layer, all layers fully connected
# '[64,32]' would be 64 nodes in first layer and 32 in second.
# feature columns is an iterable of all the feature columns in the model.



#sk learn compatability helper 
#note this is changed from the textbook, the SK compatability wrapper is changed to an import!


#from tensorflow.contrib.learn.python import SKCompat
#dnn_clf = SKCompat(dnn_clf) # if TensorFlow >= 1.1
dnn_clf.fit(train_x, train_y, batch_size=50, steps=50000)


"""
predict probabilities with the neural network
"""
dnn_y_pred = dnn_clf.predict_proba(test_dat)

dnn_out = list(dnn_y_pred)


"""
output the data
"""

dnn_output = submission
dnn_output['target'] = [x[1] for x in dnn_out]

dnn_output.to_csv('dnn_predictions5.csv', index=False, float_format='%.4f')


#try 1: hidden_units=[150,150,150] , steps=40000  ~0.268
#try 2: hidden_units=[200,150,100] , steps=100000 ~0.262
#try 3: hidden_units=[150,150,150] , steps=3000
#try 4: hidden_units=[150,150,150,150] , steps=20000 ~0.261
#try 5: hidden_units=[150,150,150,150] , steps=60000
#try 6: hidden_units=[150,150,150,100,50] , steps=50000


