# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
This code is a modified version of the code from this link:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py

His code is a very good one for RNN beginners. Feel free to check it out.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import cPickle as pickle
import matplotlib.pyplot as plt
import argparse
import math
import gzip
import time

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
f =gzip.open('./DetectionBinsData_pickle615_clean.gzip','rb')
# hyperparameters
lr = 0.001
training_iters = 10000
batch_size = 100
num_bins=100

n_inputs = 1   # MNIST data input (img shape: 28*28)
n_steps = num_bins    # time steps
n_hidden_units = 400   # neurons in hidden layer
n_classes = 2      # MNIST classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float64))

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        data=pickle.load(f)
        dd=data[:,1:num_bins+1]
        y_d=100*[[1,0]]
        bb=data[:,102:102+num_bins]
        y_b=100*[[0,1]]
        batch_xs, batch_ys = dd, y_d
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        batch_xs, batch_ys = bb, y_b
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
            }))
        step += 1

    #test
    start = time.time()
    acc_set=[]
    for p in range(10):
        error_cnt_b=0
        error_cnt_d=0
        for i in range (100):
            data=pickle.load(f)
            dd=data[:,1:num_bins+1]
            bb=data[:,102:102+num_bins]
            batch_xs = dd
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
            predict_d = sess.run(pred, feed_dict={x: batch_xs})
            for j in range(100):
                if predict_d[j][0]<=predict_d[j][1]:
                    error_cnt_d+=1
            batch_xs = bb
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
            predict_b = sess.run(pred, feed_dict={x: batch_xs})
            for j in range(100):
                if predict_b[j][0]>=predict_b[j][1]:
                    error_cnt_b+=1
        error_rate_d=(float)(error_cnt_d/10000.0)
        error_rate_b=(float)(error_cnt_b/10000.0)
        accuracy_rate=1-(float)(error_cnt_b+error_cnt_d)/20000.0
        #print('error_dark sate:',error_cnt_d,error_rate_d)
        #print('error_bright sate:',error_cnt_b,error_rate_b)
        #print('total accuracy rate:',accuracy_rate) 
        acc_set.append(accuracy_rate)    
    print( float(np.mean(acc_set)),float(np.mean(acc_set)-np.min(acc_set)),float(np.max(acc_set)-np.mean(acc_set)))
    f.close()

    end = time.time()
    print ('Time used: ',end-start)
