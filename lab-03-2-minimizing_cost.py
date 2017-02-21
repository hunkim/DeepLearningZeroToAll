# Lab 3 Minimizing Cost
import tensorflow as tf
import numpy as np

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# Try to find values for W and b to compute y_data = W * x_data + b
# We know that W should be 1 and b should be 0
# But let's use Tensorflow to figure it out
W = tf.Variable(tf.random_uniform(
    shape=[1], minval=-10.0, maxval=10.0, dtype=tf.float32))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# Cost function
cost = tf.reduce_sum(tf.square(hypothesis - Y))

# Minimize
descent = W - \
    tf.multiply(0.1, tf.reduce_mean(tf.multiply((tf.multiply(W, X) - Y), X)))
update = W.assign(descent)

# Initialize variables
init = tf.global_variables_initializer()

# Launch the graph
sess = tf.Session()
sess.run(init)

for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
