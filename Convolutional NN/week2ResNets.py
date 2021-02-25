#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 21:44:23 2021

@author: beckswu
"""

import numpy as np
import tensorflow as tf
import pydot
from IPython.display import SVG
from resnets_utils import *
import scipy.misc
from matplotlib.pyplot import imshow

import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def identity_block(X, f, filters, stage, block):
    """

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
 
    # Save the input value.  need this later to add back to the main path. 
    X_shortcut = X
    
    X = tf.keras.layers.Conv2D(filters = F1, kernel_size = (1,1), strides=(1,1), padding = 'valid',
                   name = conv_name_base + '2a',
                   kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    # Second component of main path
    X = tf.keras.layers.Conv2D(filters = F2, kernel_size =(f,f), strides = (1,1), padding='same',
                   name = conv_name_base + '2b',
                   kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    # Third component of main path
    X = tf.keras.layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1),
                   padding = 'valid', name = conv_name_base + '2c',
                   kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    
    
    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)
    
    return X


def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = tf.keras.layers.Conv2D(F1, (1, 1), strides = (s,s), 
                   name = conv_name_base + '2a', 
                   kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)


    # Second component of main path 
    X = tf.keras.layers.Conv2D(F2, (f, f), strides = (1,1), padding = 'same', 
                   name = conv_name_base + '2b',
                   kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Third component of main path 
    X = tf.keras.layers.Conv2D(F3, (1, 1), strides = (1,1), padding = 'valid', 
                   name = conv_name_base + '2c',
                   kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = tf.keras.layers.Conv2D(F3, (1, 1), strides = (s,s), padding = 'valid', 
                    name = conv_name_base + '1', 
                    kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)
    
    return X


def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    X_input = tf.keras.layers.Input(input_shape)
    # Zero-Padding
    X = tf.keras.layers.ZeroPadding2D((3,3))(X_input)
    
    # Stage 1
    X = tf.keras.layers.Conv2D(filters = 64, kernel_size = (7,7), strides = (2,2),name='conv1',
                    kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3,name = 'bn_conv1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2))(X)
    
    # Stage 2
    X = convolutional_block(X, f=3, filters=[64,64,256], stage=2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    
    # Stage 3
    X = convolutional_block(X, f=3, filters=[128,128,512], stage=3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    
    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    
    # Stage 5
    X = convolutional_block(X, f=3, filters=[512,512,2048], stage=5, block='a', s = 2)
    X = identity_block(X, 3, [512,512,2048], stage=5, block='b')
    X = identity_block(X, 3, [512,512,2048], stage=5, block='c')
    
    X = tf.keras.layers.AveragePooling2D(pool_size = (2,2), padding='same', name ='avg_pool')(X)
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(units=classes,activation='softmax', name='fc'+str(classes),
              kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)
    
    model = tf.keras.Model(inputs = X_input, outputs=X, name ='ResNet50')
    return model
    
    
model = ResNet50(input_shape = (64, 64, 3), classes = 6)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

model.fit(X_train, Y_train, epochs = 2, batch_size = 32)

preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

print(model.summary())

tf.keras.utils.plot_model(model, to_file='model.png')
SVG(tf.keras.utils.model_to_dot(model).create(prog='dot', format='svg'))

def test():
    img_path = 'images/my_image.jpg'
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
    
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.imagenet_utils.preprocess_input(x)
    print('Input image shape:', x.shape)
    my_image = scipy.misc.imread(img_path)
    imshow(my_image)
    print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
    print(model.predict(x))


"""
model = load_model('ResNet50.h5')
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
"""

"""
X =  convolutional_block(np.random.randn(3, 4, 4, 6).astype(np.float32), f = 2, filters = [2,4,6],
                    stage = 1, block='a')
print(X[0][1][1])
"""