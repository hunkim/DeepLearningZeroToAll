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
N = x_data.shape[0]
y_data = xy[:, [-1]]

# y_data has labels from 0 ~ 6
# [0, 1, 2, 3, 4, 5, 6]
print("Y has one of the following values")
print(np.unique(y_data))

# x_data.shape = (101, 16)
# 101 samples, 16 features
# y_data.shape = (101, 1)
# 101 samples, 1 label (0 ~ 6)
print("Shape of X data: ", x_data.shape)
print("Shape of Y data: ", y_data.shape)

nb_classes = 7  # 0 ~ 6

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6

Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
Y_one_hot = tf.cast(Y_one_hot, tf.float32)

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


# diff
assert y_pred.shape.as_list() == Y_one_hot.shape.as_list()

diff = (y_pred - Y_one_hot)

# Back prop (chain rule)
d_layer1 = diff
d_b = 1 * d_layer1
d_w = tf.matmul(tf.transpose(X), d_layer1) / N

# Updating network using gradients
learning_rate = 0.5
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

    for step in range(500):
        sess.run(train_step, feed_dict={X: x_data, Y: y_data})

        if step % 10 == 0:
            # At 450 Step, you should see an accuracy of 100% 
            acc = sess.run(acct_res, feed_dict={X: x_data, Y: y_data})
            print("Step: {:5}\t Acc: {:.2%}".format(step, acc))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    for p, y in zip(pred, y_data):
        if p == y[0]:
            msg = "Correct!\t Prediction: {:d}\t True Y: {:d}"
        else:
            msg = "Wrong!  \t Prediction: {:d}\t True Y: {:d}"

        print(msg.format(p, int(y[0])))

'''
Correct!     Prediction: 0   True Y: 0
Correct!     Prediction: 0   True Y: 0
Correct!     Prediction: 3   True Y: 3
Correct!     Prediction: 0   True Y: 0
Correct!     Prediction: 0   True Y: 0
Correct!     Prediction: 0   True Y: 0
Correct!     Prediction: 0   True Y: 0
Correct!     Prediction: 3   True Y: 3
Correct!     Prediction: 3   True Y: 3
Correct!     Prediction: 0   True Y: 0
Correct!     Prediction: 0   True Y: 0
'''
