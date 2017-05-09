"""A very simple MNIST classifier.
Design patter inspired by
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100
label_size = 10
image_size = 28
image_pixels = image_size * image_size

# Network Parameters
hidden1_size = 256  # 1st layer number of features
hidden2_size = 256  # 2nd layer number of features

# input place holders
images = tf.placeholder(tf.float32, [None, image_pixels])
labels = tf.placeholder(tf.float32, [None, label_size])


def multilayer_perceptron_model(x, weights, biases):
    # 1st Hidden layer with RELU activation
    hidden_layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    hidden_layer_1 = tf.nn.relu(hidden_layer_1)

    # 2nd Hidden layer with RELU activation
    hidden_layer_2 = tf.add(
        tf.matmul(hidden_layer_1, weights['h2']), biases['b2'])
    hidden_layer_2 = tf.nn.relu(hidden_layer_2)

    # Output layer with linear activation
    out_layer = tf.matmul(hidden_layer_2, weights['out']) + biases['out']
    return out_layer


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([image_pixels, hidden1_size])),
    'h2': tf.Variable(tf.random_normal([hidden1_size, hidden2_size])),
    'out': tf.Variable(tf.random_normal([hidden2_size, label_size]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([hidden1_size])),
    'b2': tf.Variable(tf.random_normal([hidden2_size])),
    'out': tf.Variable(tf.random_normal([label_size]))
}

# Construct model
model = multilayer_perceptron_model(images, weights, biases)

# Define loss and optimizer
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=model))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_loss = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_images, batch_labels = mnist.train.next_batch(batch_size)
            feed_dict = {images: batch_images, labels: batch_labels}
            loss_val, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
            avg_loss += loss_val / total_batch

        print('Epoch:', '%04d' % (epoch + 1),
              'Average loss =', '{:.9f}'.format(avg_loss))

    print('Learning Finished!')

    # Test model and check accuracy
    correct_prediction = tf.equal(
        tf.argmax(model, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={
          images: mnist.test.images, labels: mnist.test.labels}))

'''
Epoch: 0001 Average loss = 153.230835746
Epoch: 0002 Average loss = 41.691877089
Epoch: 0003 Average loss = 26.147311639
Epoch: 0004 Average loss = 18.179648803
Epoch: 0005 Average loss = 13.294155821
Epoch: 0006 Average loss = 9.906689093
Epoch: 0007 Average loss = 7.412918225
Epoch: 0008 Average loss = 5.551147055
Epoch: 0009 Average loss = 4.144373992
Epoch: 0010 Average loss = 3.219636172
Epoch: 0011 Average loss = 2.363290055
Epoch: 0012 Average loss = 1.796046966
Epoch: 0013 Average loss = 1.375669293
Epoch: 0014 Average loss = 1.050870386
Epoch: 0015 Average loss = 0.788914391
Epoch: 0016 Average loss = 0.754600729
Epoch: 0017 Average loss = 0.696869777
Epoch: 0018 Average loss = 0.487060465
Epoch: 0019 Average loss = 0.440104027
Epoch: 0020 Average loss = 0.464273822
Learning Finished!
0.9499
'''
