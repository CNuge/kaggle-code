
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def gini_tf(actual, pred):
	"""a tensorflow version of normalized gini coefficient, 
	uses in top k to calculate """
	assert (len(actual) == len(pred))
	n = int(actual.get_shape()[-1])
	indices = tf.reverse(tf.nn.top_k(pred, k=n)[1], axis=0)
	a_s = tf.gather(actual, indices)
	a_c = tf.cumsum(a_s)
	giniSum = tf.reduce_sum(a_c) / tf.reduce_sum(a_s)
	giniSum -= (n + 1) / 2.
	return giniSum / n

def gini(actual, pred, cmpcol = 0, sortcol = 1):
	assert( len(actual) == len(pred) )
	all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
	all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
	totalLosses = all[:,0].sum()
	giniSum = all[:,0].cumsum().sum() / totalLosses

	giniSum -= (len(actual) + 1) / 2.
	return giniSum / len(actual)
 
def gini_normalized(a, p):
	return gini(a, p) / gini(a, a)
 
# Load the data

test_dat = pd.read_csv('test.csv')
train_dat = pd.read_csv('train.csv')
submission = pd.read_csv('sample_submission.csv')

train_y = train_dat['target']
train_x = train_dat.drop(['target', 'id'], axis = 1)
test_dat = test_dat.drop(['id'], axis = 1)

merged_dat = pd.concat([train_x, test_dat],axis=0)

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


train_x = train_x.astype(np.float32)
test_dat = test_dat.astype(np.float32)

train_x.shape
#227 predictors


#for stability
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


reset_graph()

num_inputs = train_x.shape[1]
learning_rate = 0.01
num_classes = 2
n_hidden1 = 100
n_hidden2 = 400
n_hidden3 = 200
n_hidden4 = 100
dropout = 0.2
n_epochs = 70


X = tf.placeholder(tf.float32, shape=(None, num_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")


with tf.variable_scope('ClassNet'):

	he_init = tf.contrib.layers.variance_scaling_initializer()

	training = tf.placeholder_with_default(False, shape=(), name='training')

	hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
							  kernel_initializer=he_init, name="hidden1")

	bn1 = tf.layers.batch_normalization(hidden1, training = training, momentum = 0.9)

	hidden2 = tf.layers.dense(bn1, n_hidden2, activation=tf.nn.relu,
							  kernel_initializer=he_init, name="hidden2")

	hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu,
							  kernel_initializer=he_init, name="hidden3")

	hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu,
							  kernel_initializer=he_init, name="hidden4")

	fc1 = tf.layers.dropout(hidden4, rate=dropout)

	#logits = tf.layers.dense(fc1, num_classes, activation=tf.nn.sigmoid)
	logits = tf.layers.dense(fc1, num_classes, activation=tf.nn.relu)
	
	outputs = tf.nn.softmax(logits)
	#when outputs == sigmoid, accuracy >>> high, but poor probabilities

with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
	loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	training_op = optimizer.minimize(loss)

"""
with tf.name_scope("loss"):	
	gini = gini_tf(y, outputs)
	loss = tf.reduce_mean(gini, name="loss")

with tf.name_scope("train"):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	training_op = optimizer.minimize(loss)

"""


#this is the one to switch, want it based off of gini not the xentropy
#look at the docs to confirm this
#the training_op = optimizer.minimize(loss) below should then
#be changed to maximize(gini)


with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1)
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


init = tf.global_variables_initializer()
saver = tf.train.Saver()


X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2)


#batch_size = 100
#X_batch, y_batch = tf.train.batch([X_train, y_train], batch_size=100)
#for iteration in range(X_train.shape[0] // batch_size):		


with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		sess.run(training_op, feed_dict={X: X_train, y: y_train})	
		acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train})
		acc_test = accuracy.eval(feed_dict={X: X_val,
											y: y_val})

		###below is the new GINI test.
		prob_test = outputs.eval(feed_dict={X: X_val,
								y: y_val})
		
		gini_n = gini_normalized(y_val, prob_test[:,1])

		print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test, 
			"\nGINI NORM:", gini_n)
	
	save_path = saver.save(sess, "./cams_model_final.ckpt")




#make external predictions on the test_dat
with tf.Session() as sess:
    saver.restore(sess, "./cams_model_final.ckpt") # or better, use save_path
    Z = outputs.eval(feed_dict={X: test_dat})
    y_pred = Z[:,1]



dnn_output = submission
dnn_output['target'] = y_pred

dnn_output.to_csv('tf_dnn_predictions3.csv', index=False, float_format='%.10f')

