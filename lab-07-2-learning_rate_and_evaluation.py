# Lab 7 Learning rate and Evaluation
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder("float", [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder("float", [None, 10])

# Hypothesis (using softmax)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for step in range(2001):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / 10000)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            # Fi training using batch data
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys})
        if step % 200 == 0:
            print("Epoch: ", '%04d' % (step + 1),
                  "cost=", "{:.9f}".format(avg_cost))

    print("Optimization finished")

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(hypothesis, 1), {X: mnist.test.images[r:r + 1]}))

    plt.imshow(mnist.test.images[r:r + 1].reshape(28,
                                                  28), cmap='Greys', interpolation='nearest')
    plt.show()
