


from __future__ import print_function
import tensorflow as tf

import numpy as np
import random
import cPickle as pickle
import matplotlib.pyplot as plt
import argparse
import math
import gzip
import time
save_file='./modelthreshold.ckpt'

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)


args = parser.parse_args()
#print(tf.reduce_sum([[1,2],[3,4]],reduction_indices=[0,1]))

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    
    error=tf.reduce_sum((abs(y_pre-v_ys)))

    
    result1 = sess.run(error, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})

    return result1

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,1,1], strides=[1,2,1,1], padding='SAME')

def leakyrelu(x, alpha=0.3, max_value=None):  #alpha need set
    '''ReLU.

    alpha: slope of negative section.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x


num_bins=100  #40:  87%  60:  94%  80: 97%   90:98.0%  95:98.44%   100:  98.5%
xs = tf.placeholder(tf.float32, [None, num_bins])   # 28x28
ys = tf.placeholder(tf.float32, [None, 2])  #num_p add 1 om
keep_prob = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)


W_fc1 = weight_variable([num_bins, 240])
b_fc1 = bias_variable([240])
W_fc2 = weight_variable([240, 2])
b_fc2 = bias_variable([2])


saver = tf.train.Saver()  #define saver of the check point

h_fc1 = tf.nn.tanh(tf.matmul(tf.reshape(xs,[-1,num_bins]), W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
loss = tf.reduce_mean(tf.reduce_sum(abs(ys - prediction),
                                        reduction_indices=[1])) 
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)



fig = plt.figure()
f =gzip.open('./DetectionBinsData_pickle615_clean.gzip','rb')  #49664*100 times measurement
if args.train:
    te_dd=pickle.load(f)[:,1:1+num_bins]
    te_bb=pickle.load(f)[:,102:102+num_bins]
    fig_x=[]
    fig_y=[]
    for i in range (2500):  #training times
        try:
            #first row is cooling photon counts
            #d = pickle.load(f)[:,:101] #dark state data
            #b = pickle.load(f)[:,101:] #bright state data
            #without first row
            data=pickle.load(f)
            dd=data[:,1:1+num_bins]
            y_d=100*[[1,0]]
            bb=data[:,102:102+num_bins]
            y_b=100*[[0,1]]
            '''
            if i<1000:
                _, curr_loss = sess.run([train_step,loss], feed_dict={xs: dd, ys:y_d, keep_prob: 0.5, lr:0.005})
                _, curr_loss = sess.run([train_step,loss], feed_dict={xs: bb, ys:y_b, keep_prob: 0.5, lr:0.005})
            elif i<2000:
                _, curr_loss = sess.run([train_step,loss], feed_dict={xs: dd, ys:y_d, keep_prob: 0.5, lr:0.001})
                _, curr_loss = sess.run([train_step,loss], feed_dict={xs: bb, ys:y_b, keep_prob: 0.5, lr:0.001})
            else:
                _, curr_loss = sess.run([train_step,loss], feed_dict={xs: dd, ys:y_d, keep_prob: 0.5, lr:0.0005})
                _, curr_loss = sess.run([train_step,loss], feed_dict={xs: bb, ys:y_b, keep_prob: 0.5, lr:0.0005})
            if i % 5 == 0:
                print(i)
                error=compute_accuracy(te_dd,y_d)
                #error+=compute_accuracy(te_bb,y_b)
                curr_loss=100*curr_loss  #scale
                print(curr_loss)
                #print(sess.run(loss),feed_dict={xs: dd, ys:y_d, keep_prob: 0.5, lr:0.0001})
                fig_x.append(i)
                #fig_y.append(error)
                fig_y.append(curr_loss)
                plt.ylim(0,10)
                plt.plot(fig_x, fig_y, color='blue')
                #plt.pause(0.1)
            '''
        except EOFError:
            break
    saver.save(sess, save_file)
    #plt.ylim(0,50)
    #plt.savefig('com.png')
    #plt.show()
if args.test:
    '''
    #show the predict results
    dd=pickle.load(f)[:10,1:101]
    y_d=10*[[1,0]]
    bb=pickle.load(f)[:10,102:]
    y_b=10*[[0,1]]
    saver.restore(sess, save_file)
    predict_d = sess.run(prediction, feed_dict={xs: dd, keep_prob: 1})
    print(predict_d)
    predict_b = sess.run(prediction, feed_dict={xs: bb, keep_prob: 1})
    print(predict_b)
    '''
    start = time.time()
    num_set=[]
    y_set=[]
    y_err_set=[]
    l_err_set=[]
    u_err_set=[]
    for m in range(10,11):
        acc_set=[]
        f =gzip.open('./DetectionBinsData_pickle615_clean.gzip','rb')
        num_bins=10*m
        num_set.append(num_bins)
        for p in range(10):
            #show the error/accuracy rate
            error_cnt_b=0
            error_cnt_d=0
            #saver.restore(sess, save_file)
            test_samples=10   #test_samples * 100 *100 = test num
            threshold = 2  # 1:99.16%  2:98.5% wrong!    
            #correct is: 
            #d>=  1: 95.9%  2: 98.36%  3: 96.6
            #b<=  1: 98.36%  2: 96.7%  3: 92.9%
            for k in range (test_samples):
                for i in range (100):
                    data=pickle.load(f)
                    dd=data[:,1:1+num_bins]
                    bb=data[:,102:102+num_bins]
                    #predict_d = sess.run(prediction, feed_dict={xs: dd, keep_prob: 1})
                    d_cnt=np.sum(dd,axis=1)
                    #print(d_cnt)
                    b_cnt=np.sum(bb,axis=1)
                    #print(b_cnt)
                    for j in range(100):
                        if d_cnt[j] >= threshold:
                            error_cnt_d+=1
                    #predict_b = sess.run(prediction, feed_dict={xs: bb, keep_prob: 1})
                    for j in range(100):
                        if b_cnt[j] < threshold:
                            error_cnt_b+=1
            error_rate_d=(float)(error_cnt_d/(test_samples*100*100.0))
            error_rate_b=(float)(error_cnt_b/(test_samples*100*100.0))
            accuracy_rate=1-(float)(error_cnt_b+error_cnt_d)/(test_samples*2*100*100.0)
            #print('error_dark sate:',error_cnt_d,error_rate_d)
            #print('error_bright sate:',error_cnt_b,error_rate_b)
            #print('total accuracy rate:',accuracy_rate)
            acc_set.append(accuracy_rate)

        #print(acc_set)
        print(np.mean(acc_set),np.mean(acc_set)-np.min(acc_set),np.max(acc_set)-np.mean(acc_set))    
        y_set.append(float(np.mean(acc_set)))
        l_err_set.append(float(np.mean(acc_set)-np.min(acc_set)))
        u_err_set.append(float(np.max(acc_set)-np.mean(acc_set)))
        f.close()
    y_err_set.append(l_err_set)
    y_err_set.append(u_err_set)
    print(num_set)
    print((y_set))
    print((y_err_set))
    end = time.time()
    print ('Time used: ',end-start)