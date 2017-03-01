# http://blog.aloni.org/posts/backprop-with-tensorflow/
# https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.b3rvzhx89
# WIP
#
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
Y_one_hot = tf.cast(tf.reshape(
    Y_one_hot, [-1, nb_classes]), "float32")  # one hot

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')


def sigma(x):
    #  sigmoid function
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(-x)))


def sigma_prime(x):
    # derivative of the sigmoid function
    return sigma(x) * (1 - sigma(x))

# Forward prop
layer1 = tf.add(tf.matmul(X, W), b)
y_pred = sigma(layer1)

print(y_pred, Y_one_hot)


# diff
diff = (y_pred - Y_one_hot)

# Back prop (chain rule)
d_layer1 = diff * sigma_prime(layer1)
d_b = 1 * d_layer1
d_w = tf.matmul(tf.transpose(X), d_layer1)

# Updating network using gradients
learning_rate = 0.1
train_step = [
    tf.assign(W, W - learning_rate * d_w),
    tf.assign(b, b - learning_rate *
              tf.reduce_mean(d_b)),
]

# 7. Running and testing the training process
prediction = tf.argmax(y_pred, 1)
acct_mat = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y_one_hot, 1))
acct_res = tf.reduce_mean(tf.cast(acct_mat, tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(train_step, feed_dict={X: x_data, Y: y_data})
        print(step, sess.run(acct_res, feed_dict={X: x_data, Y: y_data}))

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
