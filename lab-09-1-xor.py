# Lab 9 XOR
# This example does not work
import tensorflow as tf
import numpy as np

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


W = tf.Variable(tf.random_uniform(
    shape=[2, 1], minval=-1.0, maxval=1.0, dtype=tf.float32))

# Hypothesis
h = tf.matmul(X, W)
hypothesis = tf.div(1., 1. + tf.exp(-h))

# Cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y)
                       * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Initialize variables
init = tf.global_variables_initializer()

# Launch graph
with tf.Session() as sess:
    sess.run(init)

    for step in range(1001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={
                  X: x_data, Y: y_data}), sess.run(W))

    # Test model
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)

    # Accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5),
                    correct_prediction, accuracy], feed_dict={X: x_data, Y: y_data}))
    print("Accuracy: ", accuracy.eval({X: x_data, Y: y_data}))
