import tensorflow as tf
import numpy as np
import cv2,os,shutil,random
import tfrecords
    
def conv_layer(x_image, W_size, weight_name, b_size, bias_name, stride, padding):
    W_conv1 = tf.Variable(tf.random_normal(W_size, stddev=0.1), name=weight_name)
    conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, stride, stride, 1], padding=padding)
    b_conv1 = tf.Variable(tf.random_normal(b_size, stddev=0.01), name=bias_name)
    h_conv1 = conv1 + b_conv1
    return h_conv1

name, sample, label = tfrecords.read_and_decode('test.tfrecords')
filename, image, y_ = tf.train.batch([name, sample, label],batch_size=1, capacity=16, num_threads=4)
y = tf.one_hot(y_, 2)

h_conv1 = conv_layer(x, [3, 3, 3, 10], 'W_conv1', [10], 'b_conv1', 1, 'SAME')
r1 = tf.nn.relu(h_conv1)

h_conv2 = conv_layer(r1, [3, 3, 10, 16], 'W_conv2', [16], 'b_conv2', 1, 'SAME')
r2 = tf.nn.relu(h_conv2)
pool2 = tf.nn.max_pool(r2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

h_conv3 = conv_layer(pool2, [3, 3, 16, 16], 'W_conv3', [16], 'b_conv3', 1, 'SAME')
r3 = tf.nn.relu(h_conv3)
pool3 = tf.nn.max_pool(r3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

h_conv4 = conv_layer(pool3, [3, 3, 16, 16], 'W_conv4', [16], 'b_conv4', 1, 'SAME')
r4 = tf.nn.relu(h_conv4)
pool4 = tf.nn.max_pool(r4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

h_conv5 = conv_layer(pool4, [3, 3, 16, 16], 'W_conv5', [16], 'b_conv5', 1, 'SAME')
r5 = tf.nn.relu(h_conv5)
pool5 = tf.nn.max_pool(r5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

h_conv6 = conv_layer(pool5, [3, 3, 16, 16], 'W_conv6', [16], 'b_conv6', 2, 'SAME')
flat = tf.reshape(h_conv6, [-1, 5*6*16])
W_fc = tf.Variable(tf.random_normal([5*6*16, 2], stddev=0.1, dtype=tf.float32), 'W_fc')
b_fc = tf.Variable(tf.random_normal([2], stddev=0.1, dtype=tf.float32), 'b_fc')
fc_add = tf.matmul(flat, W_fc) + b_fc
y_conv = tf.nn.softmax(fc_add)
pred = tf.argmax(y_conv,1)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc_add, labels=y))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy, var_list=[var for var in tf.trainable_variables()])
saver=tf.train.Saver(max_to_keep=5000, var_list=[var for var in tf.trainable_variables()])

    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    threads = tf.train.start_queue_runners(sess=sess)
    cnt = 89
    TP = 0.
    FP = 0.
    TN = 0.
    FN = 0.
    saver.restore(sess, 'checkpoints/%d.ckpt'%(cnt))            
    for i in range(100):
        img_name, predict, gt = sess.run([filename ,pred, y_])
        if predict[0] == gt[0]:
            if gt[0] == 0:
                TN += 1                
            else:
                TP += 1
        else:
            if gt[0] == 0:
                FP += 1
            else:
                FN += 1                
    SPEC = TN/(TN+FP+1e-5)
    SEN = TP/(TP+FN+1e-5)
    print('\nEpoch = %d'%cnt)
    print('TP = %g FN = %g'%(TP,FN))
    print('TN = %g FP = %g'%(TN,FP))
    print('SEN = %f'%(SEN))
    print('SPEC = %f'%(SPEC))