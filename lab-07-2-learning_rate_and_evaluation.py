# Lab 7 Learning rate and Evaluation
import tensorflow as tf
import random
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

W = tf.Variable(tf.zeros([784, nb_classes]))
b = tf.Variable(tf.zeros([nb_classes]))

# parameters
training_epochs = 15
batch_size = 100

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

# Hypothesis (using softmax)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.).minimize(cost)

# Test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")

    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

    plt.imshow(mnist.test.images[r:r + 1].reshape(28,
                                                  28), cmap='Greys', interpolation='nearest')
    plt.show()


'''
Epoch: 0001 cost = 1.653421078
Epoch: 0002 cost = 1.578180347
Epoch: 0003 cost = 1.567036718
Epoch: 0004 cost = 1.560953102
Epoch: 0005 cost = 1.557037002
Epoch: 0006 cost = 1.553987361
Epoch: 0007 cost = 1.551654525
Epoch: 0008 cost = 1.549694977
Epoch: 0009 cost = 1.548126057
Epoch: 0010 cost = 1.546662330
Epoch: 0011 cost = 1.545407041
Epoch: 0012 cost = 1.544319339
Epoch: 0013 cost = 1.543301272
Epoch: 0014 cost = 1.542362210
Epoch: 0015 cost = 1.541643034
Learning finished
Accuracy:  0.9275
Label:  [0]
Prediction:  [0]
'''
