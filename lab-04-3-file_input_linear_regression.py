# Lab 4 Multi-variable linear regression
import tensorflow as tf
import numpy as np

xy = np.loadtxt('data.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

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

# Initialize variables
init = tf.global_variables_initializer()

# Launch graph
sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost),
              sess.run(hypothesis), sess.run(W), sess.run(b))
