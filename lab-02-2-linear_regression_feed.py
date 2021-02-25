# Lab 2 Linear Regression
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# Try to find values for W and b to compute Y = W * X + b
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# placeholders for a tensor that will be always fed using feed_dict
# See http://stackoverflow.com/questions/36693740/
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# Our hypothesis is X * W + b
hypothesis = X * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    # Fit the line
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run(
            [train, cost, W, b], feed_dict={X: [1, 2, 3], Y: [1, 2, 3]}
        )
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)

    # Testing our model
    print(sess.run(hypothesis, feed_dict={X: [5]}))
    print(sess.run(hypothesis, feed_dict={X: [2.5]}))
    print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))

    # Learns best fit W:[ 1.],  b:[ 0]
    """
    0 3.5240757 [2.2086694] [-0.8204183]
    20 0.19749963 [1.5425726] [-1.0498911]
    ...
    1980 1.3360998e-05 [1.0042454] [-0.00965055]
    2000 1.21343355e-05 [1.0040458] [-0.00919707]
    [5.0110054]
    [2.500915]
    [1.4968792 3.5049512]
    """

    # Fit the line with new training data
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run(
            [train, cost, W, b],
            feed_dict={X: [1, 2, 3, 4, 5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]},
        )
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)

    # Testing our model
    print(sess.run(hypothesis, feed_dict={X: [5]}))
    print(sess.run(hypothesis, feed_dict={X: [2.5]}))
    print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))

    # Learns best fit W:[ 1.],  b:[ 1.1]
    """
    0 1.2035878 [1.0040361] [-0.00917497]
    20 0.16904518 [1.2656431] [0.13599995]
    ...
    1980 2.9042917e-07 [1.00035] [1.0987366]
    2000 2.5372992e-07 [1.0003271] [1.0988194]
    [6.1004534]
    [3.5996385]
    [2.5993123 4.599964 ]
    """
