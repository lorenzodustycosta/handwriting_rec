# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 14:56:34 2019

@author: Lorenzo
"""


import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import one_hot_encode

import tensorflow as tf
import matplotlib.pyplot as plt

def error_rate(p, t):
    return np.mean(p!=t)

def CNN_model(Xtrain, labels):
    N = int(np.sqrt(Xtrain.shape[1]))
    input_layer = tf.reshape(tf.Variable( Xtrain.astype(np.float32), name = "input_layer"),[-1,N,N,1])

    # Convolutional Layer #1 
    conv1 = tf.layers.conv2d(inputs=input_layer,filters=4,kernel_size=[5,5],padding="same", activation=tf.nn.relu)
    
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    # Convolutiona Layer #2
    conv2 = tf.layers.conv2d(inputs=pool1, filters=8, kernel_size=[5, 5], activation=tf.nn.relu)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, int(pool2.shape[1]) * int(pool2.shape[2]) * int(pool2.shape[3])])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)
    
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=27)
    
    return logits
    
        
if __name__ == "__main__":
    print("Starting")
   
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"Training")
    
    train = pd.read_csv(path + "/emnist-letters-train/emnist-letters-train.csv", header = None)        
    test = pd.read_csv(path + "/emnist-letters-test/emnist-letters-test.csv", header = None)        
    print("Data Loaded")
    
    df = train.append(test)
    df = shuffle(df)
    
    X_df = df.iloc[:,1:].values
    X_df[X_df >0] = 1
    y_df = df.iloc[:,0].values

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_df, y_df, test_size=0.20, random_state=42)
    print("Split")

    
    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N // batch_sz
    K = 27
    Ytrain_ind = one_hot_encode(ytrain, K)
    Ytest_ind = one_hot_encode(ytest, K)
      
    logits = CNN_model(Xtrain, ytrain)  
    print("CNN Model created")
 
     # define placeolder for input and output
    X = tf.placeholder(tf.float32, shape = (None,D), name='X')
    Y = tf.placeholder(tf.float32, shape = (None,K), name='Y')

    # define cost, training and prediction operators        
    cost_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y))
    train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost_op)
    predicts_op = tf.argmax(logits, axis = 1)
    print("Cost, train and predictions defined")
     
    # initialize tf session
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    print("Start TF session")
 

    max_iter = 20
    print_period = 10
    LL = []
    EE = []
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[ j*batch_sz : (j+1)*batch_sz , ]
            Ybatch = Ytrain_ind[ j*batch_sz : (j+1)*batch_sz , ]
            
            sess.run(train_op, feed_dict = {X: Xbatch, Y: Ybatch})
            
            #cost_val = sess.run(cost_op, feed_dict={X: Xtest, Y: Ytest_ind})
            prediction = sess.run(predicts_op, feed_dict={X: Xtest})
            
            err = error_rate(prediction, ytest)
            
           #.. LL.append(cost_val)
            EE.append(err)
            
            if j % print_period == 0:
                print("Cost / err at iteration i=%d, j=%d: %.0f / %.3f" %(i,j,0,err))
    
    #plt.figure()
    #plt.plot(LL)
    plt.figure()
    plt.plot(EE)    
    
#    saver = tf.train.Saver()
#    saver.save(sess, "model")
#    
#    sess.close() 
