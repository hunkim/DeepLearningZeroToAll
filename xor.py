import tensorflow as tf
import numpy as np

x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [[0], [1], [1], [0]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.random_uniform([2,2],-1.0,1.0))
b1 = tf.Variable(tf.zeros([2]), name = "Bias1")
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_uniform([2,1],-1.0,1.0))
b2 = tf.Variable(tf.zeros([1]), name = "Bias2")
H = tf.sigmoid(tf.matmul(layer1,W2) + b2)

cost = -tf.reduce_mean(Y*tf.log(H) + (1-Y)*tf.log(1-H))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(0,20000):
        sess.run(train, feed_dict = {X:x_data, Y:y_data})
        if step%20 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data,Y:y_data}),sess.run(W1),sess.run(W2))

    correct_prediction = tf.equal(tf.floor(H+0.5),Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print("\nFINAL RESULT")
    print(sess.run( [H, tf.floor(H+0.5), correct_prediction, accuracy] , feed_dict = {X:x_data,Y:y_data}))
    print("Accuracy:", accuracy.eval({X:x_data, Y:y_data}))