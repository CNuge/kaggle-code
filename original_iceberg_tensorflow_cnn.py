#try: up the iterations and the learning rate!

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split



"""
https://www.tensorflow.org/api_docs/python/tf/layers
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
This example is using TensorFlow layers API


TensorFlow’s high-level machine learning API (tf.estimator) makes it easy to configure, 
train, and evaluate a variety of machine learning models:
https://www.tensorflow.org/get_started/estimator
"""
#####
# Load in the data
#####
print('loading data')
# load function from: https://www.kaggle.com/kmader/exploring-the-icebergs-with-skimage-and-keras
# b/c I didn't want to reinvent the wheel
def load_and_format(in_path):
	""" take the input data in .json format and return a df with the data and an np.array for the pictures """
	out_df = pd.read_json(in_path)
	out_images = out_df.apply(lambda c_row: [np.stack([c_row['band_1'],c_row['band_2']], -1).reshape((75,75,2))],1)
	out_images = np.stack(out_images).squeeze()
	return out_df, out_images


train_df, train_images = load_and_format('train.json')

test_df, test_images = load_and_format('test.json')

X_train, X_test, y_train, y_test = train_test_split(train_images,
                                                   train_df['is_iceberg'].as_matrix(),
                                                   test_size = 0.3
                                                   )
print('Train', X_train.shape, y_train.shape)
print('Validation', X_test.shape, y_test.shape)


##########################
"""
dummy_dat = np.zeros((1122,75,75,1), dtype=np.float32)

fudge_X_train = np.concatenate((X_train, dummy_dat), axis = 3)

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(fudge_X_train)

x_batches = fudge_X_train
y_batches = y_train

epochs = 10

for e in range(epochs):
    print('Epoch', e)
    batches = 0
    per_batch = 64
    for x_batch, y_batch in datagen.flow(fudge_X_train, y_train, batch_size=per_batch):
        x_batches = np.concatenate((x_batches, x_batch), axis = 0)
        y_batches = np.concatenate((y_batches, y_batch), axis = 0)
        batches += 1
        if batches >= len(fudge_X_train) / per_batch:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break



x_train_new = x_batches[:,:,:,:2]
x_train_new.shape
y_batches.shape

X_train, y_train = x_train_new, y_batches

"""
#######################


#convert to np.float32 for use in tensorflow
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)


#for stability
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


reset_graph()

print('designing model')
# Training Parameters
learning_rate = 0.005
n_epochs = 2500


# Network Parameters
num_input = 75*75 #size of the images
num_classes = 2 # Binary
dropout = 0.4 # Dropout, probability to keep units



X = tf.placeholder(tf.float32, shape=(None, 75, 75, 2), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")


with tf.variable_scope('ConvNet'):

    he_init = tf.contrib.layers.variance_scaling_initializer()

    # Convolution Layer with 32 filters and a kernel size of 5
    conv1 = tf.layers.conv2d(X, filters=32,  kernel_size=[5, 5], activation=tf.nn.relu)
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(pool1, filters=64,  kernel_size=[3,3], activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(pool2, filters=128, kernel_size=[3,3], activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2)

    conv4 = tf.layers.conv2d(pool3, filters=256, kernel_size=[3,3], activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=2)
    
    # Flatten the data to a 1-D vector for the fully connected layer
    fc1 = tf.contrib.layers.flatten(pool4)

    # Fully connected layer (in tf contrib folder for now)
    fc2 = tf.layers.dense(fc1, 32, 
                        kernel_initializer=he_init, activation=tf.nn.relu)

    fc3 = tf.layers.dense(fc2, 128, 
                        kernel_initializer=he_init, activation=tf.nn.relu)

    fc4 = tf.layers.dense(fc3, 512, 
                        kernel_initializer=he_init, activation=tf.nn.relu)

    fc5 = tf.layers.dense(fc4, 32, 
                        kernel_initializer=he_init, activation=tf.nn.relu)


    # Apply Dropout (if is_training is False, dropout is not applied)
    fc6 = tf.layers.dropout(fc5, rate=dropout)

    logits = tf.layers.dense(fc6, num_classes, activation=tf.nn.sigmoid)

    #outputs = tf.nn.softmax(logits)

#when I removed the output layer and made the logits the output, then it worked well!
#I think the issue was I was being silly and not training the outputs layer so it made
#inaccurate predictions, have switched this and it looks to be working better.

#check the output to see if the probabilities seem reasonable and then beef up the iterations.



"""
    # Output layer, class prediction
    #out = tf.nn.softmax(fc1, n_classes)
    logits = tf.layers.dense(fc6, num_classes, activation=tf.nn.sigmoid) #try to add sigmoid: , activation='sigmoid' ; softmax: ,  activation = 'softmax' 

    outputs = tf.nn.softmax(logits)
"""

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


init = tf.global_variables_initializer()
saver = tf.train.Saver()


print('training model\n')
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        sess.run(training_op, feed_dict={X: X_train, y: y_train})   
        acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train})
        acc_test = accuracy.eval(feed_dict={X: X_test,
                                            y: y_test})
    
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./iceberg_model_final.ckpt")



#convert the test images to float32
test_images =test_images.astype(np.float32) 
test_images.shape


print('making predictions\n')
#make external predictions on the test_dat
with tf.Session() as sess:
    saver.restore(sess, "./iceberg_model_final.ckpt") # or better, use save_path
    Z = logits.eval(feed_dict={X: test_images}) #outputs switched to logits
    y_pred = Z[:,1]


#below we select the probability of a '1' not prob of a '0'


output = pd.DataFrame(test_df['id'])

output['is_iceberg'] = y_pred


output.to_csv('cam_tf_cnn.csv', index=False)
