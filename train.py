# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 23:42:13 2019

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


if __name__ == "__main__":
    
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"Dataset")
    
    train = pd.read_csv(path + "/emnist-letters-train/emnist-letters-train.csv", header = None)        
    test = pd.read_csv(path + "/emnist-letters-test/emnist-letters-test.csv", header = None)        
    
    df = train.append(test)
    df = shuffle(df)
    
    X_df = df.iloc[:,1:].values
    X_df[X_df >0] = 1
    y_df = df.iloc[:,0].values

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_df, y_df, test_size=0.20, random_state=42)
    
    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N // batch_sz
    
    M1 = 1024
    M2 = 256
    K = 26
    Ytrain_ind = one_hot_encode(ytrain, K)
    Ytest_ind = one_hot_encode(ytest, K)
               
    # initialize variable    
    W1_init = np.random.rand(D, M1) / np.sqrt(D)
    b1_init = np.zeros(M1)
    W2_init = np.random.rand(M1, M2) / np.sqrt(M1)
    b2_init = np.zeros(M2)
    W3_init = np.random.rand(M2, K) / np.sqrt(M2)
    b3_init = np.zeros(K)    
        
    # define placeolder for input and output
    X = tf.placeholder(tf.float32, shape = (None,D), name='X')
    Y = tf.placeholder(tf.float32, shape = (None,K), name='Y')
    
    # define updatable variable
    W1 = tf.Variable( W1_init.astype(np.float32), name = "W1")
    b1 = tf.Variable( b1_init.astype(np.float32), name = "b1")
    W2 = tf.Variable( W2_init.astype(np.float32), name = "W2")
    b2 = tf.Variable( b2_init.astype(np.float32), name = "b2")
    W3 = tf.Variable( W3_init.astype(np.float32), name = "W3")
    b3 = tf.Variable( b3_init.astype(np.float32), name = "b3")

    # define the NN (3 layers)
    Z1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    Z2 = tf.nn.sigmoid(tf.matmul(Z1, W2) + b2)
    Yish = tf.matmul(Z2, W3) + b3 # remember, the cost function does the softmaxing! weird, right?

    
    # define cost, training and prediction operators
    cost_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Yish, labels = Y))
    #train_op = tf.train.RMSPropOptimizer(0.00004, decay = 0.99, momentum = 0.9).minimize(cost_op)    
    train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost_op) 
    predicts_op = tf.argmax(Yish, 1)

    # initialize tf session
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    max_iter = 20
    print_period = 10
    LL = []
    EE = []
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[ j*batch_sz : (j+1)*batch_sz , ]
            Ybatch = Ytrain_ind[ j*batch_sz : (j+1)*batch_sz , ]
            
            sess.run(train_op, feed_dict = {X: Xbatch, Y: Ybatch})
            
            cost_val = sess.run(cost_op, feed_dict={X: Xtest, Y: Ytest_ind})
            prediction = sess.run(predicts_op, feed_dict={X: Xtest})
            
            err = error_rate(prediction, ytest)
            
            LL.append(cost_val)
            EE.append(err)
            
            if j % print_period == 0:
                print("Cost / err at iteration i=%d, j=%d: %.0f / %.3f" %(i,j,cost_val,err))
    
    plt.figure()
    plt.plot(LL)
    plt.figure()
    plt.plot(EE)    
    
    saver = tf.train.Saver()
    saver.save(sess, "./Trained Models/model_NN")
    
    sess.close() 
