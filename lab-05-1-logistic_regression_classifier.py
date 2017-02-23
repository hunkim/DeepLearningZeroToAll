# Lab 5 Logistic Regression Classifier
import tensorflow as tf
import numpy as np

xy = np.loadtxt('data.csv', delimiter=',')
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform(
    shape=[3, 1], minval=-1.0, maxval=1.0, dtype=tf.float32))
# Hypothesis
h = tf.matmul(X, W)
hypothesis = tf.div(1., 1. + tf.exp(-h))

# cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y)
                       * tf.log(1 - hypothesis))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cost)

# Initialize variable
init = tf.global_variables_initializer()

# Launch graph
sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={
              X: x_data, Y: y_data}), sess.run(W))
