#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 14:05:50 2021

@author: beckswu
"""


import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *



X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T


# Example of a picture
index = 6
plt.imshow(X_train_orig[index])
plt.show()
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))



#tf.placeholder is used to feed actual training examples.
# tf.Variable for trainable variables such as weights (W) and biases (B) for your model.


"""
Implement the function below to create placeholders for the input image X and the output Y.
 You should not define the number of training examples for the moment. To do so,
 you could use "None" as the batch size, it will give you the flexibility to choose it later. 
 Hence X should be of dimension [None, n_H0, n_W0, n_C0] and Y should be of dimension 
     [None, n_y]. 


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.compat.v1.placeholder(dtype = tf.float32, shape = [None, n_H0, n_W0, n_C0])
    Y = tf.compat.v1.placeholder(dtype = tf.float32, shape = [None, n_y])
    return X, Y
"""
def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]

    """
    tf.random.set_seed(1)

    #using xavier_initializer
    initializer=tf.initializers.GlorotUniform(seed = 0)
    W1 = tf.Variable(initializer([4, 4, 3, 8]),name="W1")
    
    initializer=tf.initializers.GlorotUniform(seed = 0)
    W2 = tf.Variable(initializer([2, 2, 8, 16]),name="W2")
    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    """
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    # CONV2D:
    # filter order must be  [filter_height, filter_width, in_channels, out_channels]
    # input order must be [batch_shape, in_height, in_width, in_channels]
    Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding="SAME")
    print("Z1.shape ",Z1.shape)
    A1 = tf.nn.relu(Z1)
    #Max pool
    P1 = tf.nn.max_pool(A1, ksize=[1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    print("P1.shape ",P1.shape)
    # CONV2D:
    Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = "SAME")
    A2 = tf.nn.relu(Z2)
    print("A2.shape ",A2.shape)
    #Max pool
    P2 = tf.nn.max_pool(A2,ksize = [1,4,4,1], strides = [1,4,4,1], padding = "SAME")
    print("P2.shape ",P2.shape)
    #FLATTEN
    P2 = tf.keras.layers.Flatten()(P2)
    print("P2.shape ",P2.shape)
    Z3 = tf.keras.layers.Dense(6)(P2)
    
    return Z3
    

def CNNmodel(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):
    
    costs = []
    accuracy = []
    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 10 == 0:
                print("Training: end of batch {}; loss: {}, accuracy:{}; val_loss: {}, val_accuracy:{}".format(epoch,
                      round(logs['loss'],4),round(logs['accuracy'],4),round(logs['val_loss'],4),round(logs['val_accuracy'],4)))
            costs.append(logs['loss'])
            accuracy.append(logs['accuracy'])
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(8,(4,4),padding = 'SAME', activation = tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D(8,8))
    
    model.add(tf.keras.layers.Conv2D(16,(2,2),padding = 'SAME', activation = tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D(4,4))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(6, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, 
                  metrics=["accuracy"])
   
    model.fit(X_train, Y_train,batch_size=minibatch_size, epochs=num_epochs,verbose=0,
             callbacks=[MyCallback()], validation_data=(X_test, Y_test))
    
    print(model.summary())
    plt.plot(np.squeeze(costs), 'g-', label='Loss')
    plt.plot(np.squeeze(accuracy), 'blue', label='Accuracy')
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.legend(loc="upper right")
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    model.evaluate(X_train, Y_train)

    model.evaluate(X_test, Y_test)
    
    
parameters = initialize_parameters()
print("W1 = " + str(parameters["W1"][1,1,1]))
print("W2 = " + str(parameters["W2"][1,1,1]))

print(forward_propagation(np.random.randn(2,64,64,3).astype(np.float32), parameters))

CNNmodel(X_train, Y_train, X_test, Y_test)




