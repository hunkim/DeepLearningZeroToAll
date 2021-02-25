# http://blog.aloni.org/posts/backprop-with-tensorflow/
# https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.b3rvzhx89
# WIP
import tensorflow as tf

tf.set_random_seed(777)  # reproducibility

# tf Graph Input
x_data = [[1.],
          [2.],
          [3.]]
y_data = [[1.],
          [2.],
          [3.]]


# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 1])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# Set wrong model weights
W = tf.Variable(tf.truncated_normal([1, 1]))
b = tf.Variable(5.)

# Forward prop
hypothesis = tf.matmul(X, W) + b

# diff
assert hypothesis.shape.as_list() == Y.shape.as_list()
diff = (hypothesis - Y)

# Back prop (chain rule)
d_l1 = diff
d_b = d_l1
d_w = tf.matmul(tf.transpose(X), d_l1)

print(X, W, d_l1, d_w)

# Updating network using gradients
learning_rate = 0.1
step = [
    tf.assign(W, W - learning_rate * d_w),
    tf.assign(b, b - learning_rate * tf.reduce_mean(d_b)),
]

# 7. Running and testing the training process
RMSE = tf.reduce_mean(tf.square((Y - hypothesis)))

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    print(i, sess.run([step, RMSE], feed_dict={X: x_data, Y: y_data}))

print(sess.run(hypothesis, feed_dict={X: x_data}))
