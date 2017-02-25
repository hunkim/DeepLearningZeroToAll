# Lab 4 Multi-variable linear regression
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # reproducibility

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch graph
sess = tf.Session()
# Initialize TensorFlow variables
sess.run(tf.global_variables_initializer())

x_data = np.array([[73, 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]])
y_data = np.array([[152.], [185.], [180.], [196.], [142.]])


for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y: y_data})
    if step % 10 == 0:
        print(step, sess.run([cost, hypothesis], feed_dict={X:x_data, Y: y_data}))

# Ask my score
score = np.array([[100, 70, 101]])
print("Your score will be ", sess.run(hypothesis, feed_dict={X:score}))

score = np.array([[60, 70, 110], [90, 100, 80]])
print("Other scores will be ", sess.run(hypothesis, feed_dict={X:score}))
