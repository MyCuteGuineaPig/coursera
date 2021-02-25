#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 12:52:57 2021

@author: beckswu
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from tensorflow.keras.utils import plot_model, model_to_dot

import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow 
from kt_utils import *

"""
In particular, rather than creating and assigning a new variable on each step of forward propagation 
such as X, Z1, A1, Z2, A2, etc. for the computations for the different layers, 

in Keras code each line above just reassigns X to a new value using X = .... 
In other words, each step of forward propagation, writing the latest value in the commputation into the same variable X.
The only exception was X_input, which we kept separate and did not overwrite, 
since we needed it at the end to create the Keras model instance (model = Model(inputs = X_input, ...) above).

"""

def HappyModel(input_shape):
    
    X_input = tf.keras.layers.Input(input_shape)
    
    # Zero-Padding: pads the border of X_input with zeroes
    X = tf.keras.layers.ZeroPadding2D((3,3,))(X_input)
    
    # CONV -> BN -> RELU Block applied to X
    X = tf.keras.layers.Conv2D(32, (7,7), strides = (1,1), name = 'conv0')(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = 'bn0')(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    #MAX Pool
    X = tf.keras.layers.MaxPool2D((2,2),name='max_pool')(X)
    
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(1, activation='sigmoid', name = 'fc')(X)
    
    model = tf.keras.models.Model(inputs = X_input, outputs = X, name = 'HappyModel')
    
    return model

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


happyModel = HappyModel(X_train.shape[1:])
happyModel.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
happyModel.fit(X_train, Y_train, epochs=40, batch_size=50)


preds = happyModel.evaluate(X_test, Y_test, batch_size = 32, verbose =1, sample_weight = None)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))




img_path = 'images/my_image.jpg'
### END CODE HERE ###
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
imshow(img)

x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happyModel.predict(x))

print(happyModel.summary())

plot_model(happyModel, to_file='HappyModel.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))