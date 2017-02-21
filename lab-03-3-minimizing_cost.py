# Lab 3 Minimizing Cost
import tensorflow as tf

# tf Graph Input
X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_samples = len(X)

# Set model weights
W = tf.Variable(0.)
# Linear model
hypothesis = tf.multiply(X, W)

# Cost function
cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2)) / (m)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# Initialize variables
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# Set model weights
wOp = W.assign(5.)
sess.run(wOp)
for step in range(10):
    print(step, sess.run(W))
    sess.run(train)
