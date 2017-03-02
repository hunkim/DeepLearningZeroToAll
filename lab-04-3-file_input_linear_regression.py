# Lab 4 Multi-variable linear regression
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Make sure the sape and data are OK
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    feed = {X: x_data, Y: y_data}
    sess.run(train, feed_dict=feed)
    if step % 10 == 0:
        print(step, "Cost: ", sess.run(cost, feed_dict=feed),
              "\nPrediction:\n", sess.run(hypothesis, feed_dict=feed))

# Ask my score
score = np.array([[100, 70, 101]])
print("Your score will be ", sess.run(hypothesis, feed_dict={X: score}))

score = np.array([[60, 70, 110], [90, 100, 80]])
print("Other scores will be ", sess.run(hypothesis, feed_dict={X: score}))

'''
Your score will be  [[ 181.73277283]]
Other scores will be  [[ 145.86265564]
 [ 187.23129272]]

'''
