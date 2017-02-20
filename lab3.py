#Lab 3 Minimizing Cost
import tensorflow as tf
import numpy as np

X = [1.,2.,3.]
Y = [1.,2.,3.]
m = n_samples = len(X)

W = tf.placeholder(tf.float32)

#Our hypothesis for linear model X * W
hypothesis = tf.multiply(X, W)

#Cost function
cost = tf.reduce_sum(tf.pow(hypothesis-Y, 2), keep_dims=False)/(m)

#Initialize variables
init = tf.global_variables_initializer()

#For graphs
W_val = []
cost_val = []

#Launch the graph
sess = tf.Session()
sess.run(init)
for i in range(-30, 50):
	print(i*0.1, sess.run(cost, feed_dict={W: i*0.1}))
	W_val.append(i*0.1)
	cost_val.append(sess.run(cost, feed_dict={W: i*0.1}))