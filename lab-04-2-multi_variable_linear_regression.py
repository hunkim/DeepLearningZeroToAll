# Lab 4 Multi-variable linear regression
import tensorflow as tf
import numpy as np

x_data = np.array([[1, 2, 1], [1, 3, 2], [1, 3, 4],
                   [1, 5, 5], [1, 7, 5], [1, 2, 5]], dtype=np.float32)
y_data = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)

W = tf.Variable(tf.random_uniform(
    shape=[3, 1], minval=-1.0, maxval=1.0, dtype=tf.float32))
b = tf.Variable(tf.random_uniform(
    shape=[1], minval=-1.0, maxval=1.0, dtype=tf.float32))

# Hypothesis
hypothesis = tf.matmul(x_data, W) + b

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Initialize variable
init = tf.global_variables_initializer()

# Launch graph
sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))