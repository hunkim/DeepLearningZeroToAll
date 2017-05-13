# Lab 9 XOR-back_prop
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.1

x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]
x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
b1 = tf.Variable(tf.random_normal([2]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

def sigmoidGradient(z):
  return tf.multiply(tf.sigmoid(z), (1 - tf.sigmoid(z)))

diff = hypothesis - Y

d_l2 = tf.multiply(diff, sigmoidGradient(tf.matmul(layer1, W2) + b2))
d_b2 = d_l2
d_W2 = tf.matmul(tf.transpose(layer1), d_l2)

d_l1 = tf.multiply(tf.matmul(d_l2, tf.transpose(W2)), sigmoidGradient(tf.matmul(X, W1) + b1))
d_b1 = d_l1
d_W1 = tf.matmul(tf.transpose(X), d_l1)

step = [
  tf.assign(W2, W2 - learning_rate * d_W2),
  tf.assign(b2, b2 - learning_rate * tf.reduce_mean(d_b2, axis=[0])),
  tf.assign(W1, W1 - learning_rate * d_W1),
  tf.assign(b1, b1 - learning_rate * tf.reduce_mean(d_b1, axis=[0]))
]

# Accuracy computation
# True if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for i in range(10001):
        sess.run([step, cost], feed_dict={X: x_data, Y: y_data})
        if i % 1000 == 0:
            print(i, sess.run(cost, feed_dict={
                  X: x_data, Y: y_data}), sess.run([W1, W2]))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)


'''
Hypothesis:  [[ 0.07884014]
 [ 0.88706875]
 [ 0.94088489]
 [ 0.04933683]]
Correct:  [[ 0.]
 [ 1.]
 [ 1.]
 [ 0.]]
Accuracy:  1.0
'''
