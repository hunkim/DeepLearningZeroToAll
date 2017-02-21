# Lab 3 Minimizing Cost
import tensorflow as tf

# tf Graph Input
X = [1, 2, 3]
Y = [1, 2, 3]

# Set model weights
W = tf.Variable(5.)

# Linear model
hypothesis = X * W

# Cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Initialize variables
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(10):
    print(step, sess.run(W))
    sess.run(train)
