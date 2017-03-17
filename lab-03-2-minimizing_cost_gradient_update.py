# Lab 3 Minimizing Cost
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# Try to find values for W and b to compute y_data = W * x_data + b
# We know that W should be 1 and b should be 0
# But let's use TensorFlow to figure it out
W = tf.Variable(tf.random_normal([1]), name='weight')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent using derivative: W -= learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

'''
0 5.81756 [ 1.64462376]
1 1.65477 [ 1.34379935]
2 0.470691 [ 1.18335962]
3 0.133885 [ 1.09779179]
4 0.0380829 [ 1.05215561]
5 0.0108324 [ 1.0278163]
6 0.00308123 [ 1.01483536]
7 0.000876432 [ 1.00791216]
8 0.00024929 [ 1.00421977]
9 7.09082e-05 [ 1.00225055]
10 2.01716e-05 [ 1.00120032]
11 5.73716e-06 [ 1.00064015]
12 1.6319e-06 [ 1.00034142]
13 4.63772e-07 [ 1.00018203]
14 1.31825e-07 [ 1.00009704]
15 3.74738e-08 [ 1.00005174]
16 1.05966e-08 [ 1.00002754]
17 2.99947e-09 [ 1.00001466]
18 8.66635e-10 [ 1.00000787]
19 2.40746e-10 [ 1.00000417]
20 7.02158e-11 [ 1.00000226]
'''
