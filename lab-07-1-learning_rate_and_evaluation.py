# Lab 7 Learning rate and Evaluation
import tensorflow as tf
import numpy as np

x_data = np.array([[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5],
                   [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]], dtype=np.float32)
y_data = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
                   [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float32)

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])

W = tf.Variable(tf.zeros([3, 3]))

# Softmax
hypothesis = tf.nn.softmax(tf.matmul(X, W))
# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)

# Cross entropy cost
cost = tf.reduce_mean(-tf.reduce_sum(Y *
                                     tf.log(hypothesis), axis=1))

# Changed learning_rate to 10
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=.1).minimize(cost)  # Change learning rate to 10.

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={
                  X: x_data, Y: y_data}), sess.run(W))

    # Evaluation our model using test dataset
    x_test = np.array([[2, 1, 1], [3, 1, 2], [3, 3, 4]], dtype=np.float32)
    y_test = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)

    # Test model
    correct_prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
        X: x_test, Y: y_test}))
