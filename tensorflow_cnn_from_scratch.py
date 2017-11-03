#To Do:
#submit sub6 if it gives a 0.29 or less we can use that to write the kernel
#with the kernel written try the following.
#1 take the data augmentation and add that to see if it improves the score
#2 write an XGboost that takes the dataframe data and the nn probabilities as input and
#makes predictions based on that. then use it to predict proba and see if the values improve
# FOR 2. need to impute the missing values in the table. Try mediam to start with.


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
train, and evaluate a variety of machine learning models

#https://www.tensorflow.org/get_started/estimator
"""
#####
# Load in the data
#####

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

#This sample is an interesting alternative to .head() that I usually use
train_df.sample(3)


# also from https://www.kaggle.com/kmader/exploring-the-icebergs-with-skimage-and-keras
X_train, X_test, y_train, y_test = train_test_split(train_images,
                                                   train_df['is_iceberg'].as_matrix(),
                                                    random_state = 1738,
                                                    test_size = 0.2
                                                   )
print('Train', X_train.shape, y_train.shape)
print('Validation', X_test.shape, y_test.shape)

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


# Training Parameters
learning_rate = 0.001
num_steps = 100
batch_size = 128
n_epochs = 3


# Network Parameters
num_input = 75*75 # MNIST data input (img shape: 28*28)
num_classes = 2 # Binary
dropout = 0.75 # Dropout, probability to keep units


# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a array
        x = x_dict['images']

        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 75, 75, 2])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        conv3 = tf.layers.conv2d(conv1, 128, 3, activation=tf.nn.relu)
        conv3 = tf.layers.max_pooling2d(conv2, 2, 2)

        conv4 = tf.layers.conv2d(conv1, 256, 3, activation=tf.nn.relu)
        conv4 = tf.layers.max_pooling2d(conv2, 2, 2)

        conv5 = tf.layers.conv2d(conv1, 512, 3, activation=tf.nn.relu)
        conv5 = tf.layers.max_pooling2d(conv2, 2, 2)

        conv6 = tf.layers.conv2d(conv1, 1024, 3, activation=tf.nn.relu)
        conv6 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 2048)
 
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout)

        # Output layer, class prediction
        #out = tf.nn.softmax(fc1, n_classes)
        out = tf.layers.dense(fc1, n_classes) #try to add sigmoid: , activation='sigmoid' ; softmax: ,  activation = 'softmax' 

    return out



# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False)
    logits_test = conv_net(features, num_classes, dropout, reuse=True)

    # Predictions : the first is not used here.
    # second line uses softmax or sigmoid to predict probabilities
    pred_classes = tf.argmax(logits_test, axis=1) #make switch for just classification
    pred_proba =  tf.nn.sigmoid(logits_test)
    #pred_proba = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_proba)
        # return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs



# Build the Estimator
model = tf.estimator.Estimator(model_fn)

num_steps = 100
batch_size = 64
n_epochs = 2


# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
     x={'images': X_train}, y=y_train,
    batch_size=batch_size, num_epochs=None, shuffle=True)


# Evaluate the Model
# Define the input function for evaluating
valid_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': X_test}, y=y_test,
    batch_size=batch_size, shuffle=False)


#loop through the epochs, evaluating the test accuracy each time.
#dial back to optimumn testing accuracy or up epochs further if last == best
for epoch in range(n_epochs):
    model.train(input_fn, steps=num_steps)
    e = model.evaluate(valid_fn)
    print("\n\nTesting Accuracy:", e['accuracy'])
    print("\n\n")


#convert the test images to float32
test_images =test_images.astype(np.float32) 
test_images.shape

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"images": test_images},
      num_epochs=1,
      shuffle=False)


# test_images prediction
# predict is calling softmax in the background (defined in model_fn)
# this returns a generator that gives probabilities of 0 and probabilites of 1
predictions = list(model.predict(input_fn=predict_input_fn))


#below we select the probability of a '1' not prob of a '0'
pred_out = [x[1] for x in predictions]

test_df.head()

output = pd.DataFrame(test_df['id'])

output['is_iceberg'] = pred_out


output.to_csv('cam_tf_cnn6.csv', index=False)

#for try 3 the proba is switched from softmax to sigmoid ot see if results improve
