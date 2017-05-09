import numpy as np
import tensorflow as tf

x_train = np.reshape(np.array([1, 2, 3, 4]), [4, 1])
y_train = np.reshape(np.array([0, -1, -2, -3]), [4, 1])

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Model input and output
x = tf.placeholder(tf.float32, [4, 1])
y = tf.placeholder(tf.float32, [4, 1])

linear_model = x * W + b

# cost/loss function with vectorized form(same as lms)
loss = tf.matmul(linear_model - y, linear_model - y, transpose_a=True)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # reset values to wrong
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
