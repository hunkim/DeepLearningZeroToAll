# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

# Predicting animal type based on various features
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

nb_classes = 7  # 1 ~ 7

X = tf.placeholder("float", [None, 16])
Y = tf.placeholder("int32", [None, 1])  # 1 ~ 7
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print(Y_one_hot)

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost
# cost = tf.reduce_mean(-tf.reduce_sum(Y *
#        tf.log(hypothesis), axis=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=Y_one_hot, logits=hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    for p, y in zip(pred, y_data):
        print("prediction: ", p, " true Y: ", y)

'''
prediction:  2  true Y:  [ 3.]
prediction:  4  true Y:  [ 4.]
prediction:  1  true Y:  [ 1.]
prediction:  1  true Y:  [ 1.]
prediction:  2  true Y:  [ 2.]
prediction:  1  true Y:  [ 1.]
prediction:  6  true Y:  [ 6.]
prediction:  1  true Y:  [ 1.]
prediction:  0  true Y:  [ 7.]
prediction:  2  true Y:  [ 2.]
'''
