# Lab 4 Multi-variable linear regression
import tensorflow as tf

x1_data = [1, 0, 3, 0, 5]
x2_data = [0, 2, 0, 4, 0]
y_data = [1, 2, 3, 4, 5]

w1 = tf.Variable(tf.random_uniform(
    shape=[1], minval=-1.0, maxval=1.0, dtype=tf.float32))
w2 = tf.Variable(tf.random_uniform(
    shape=[1], minval=-1.0, maxval=1.0, dtype=tf.float32))

b = tf.Variable(tf.random_uniform(
    shape=[1], minval=-1.0, maxval=1.0, dtype=tf.float32))

hypothesis = x1_data * w1 + x2_data * w2 + b
print(hypothesis)

# Cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Initialize variables
init = tf.global_variables_initializer()

# Launch graph
sess = tf.Session()
sess.run(init)

# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print("%s w1: %s w2: %s b: %s cost: %s" %
              (step, sess.run(w1), sess.run(w2), sess.run(b), sess.run(cost)))
