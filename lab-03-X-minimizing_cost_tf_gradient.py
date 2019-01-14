# Lab 3 Minimizing Cost
# This is optional
import tensorflow as tf

# tf Graph Input
X = [1, 2, 3]
Y = [1, 2, 3]

# Set wrong model weights
W = tf.Variable(5.)

# Linear model
hypothesis = X * W

# Manual gradient
gradient = tf.reduce_mean((W * X - Y) * X) * 2

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# Get gradients
gvs = optimizer.compute_gradients(cost)

# Optional: modify gradient if necessary
# gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]

# Apply gradients
apply_gradients = optimizer.apply_gradients(gvs)

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(101):
        gradient_val, gvs_val, _ = sess.run([gradient, gvs, apply_gradients])
        print(step, gradient_val, gvs_val)

'''
0 37.333332 [(37.333336, 5.0)]
1 33.84889 [(33.84889, 4.6266665)]
2 30.689657 [(30.689657, 4.2881775)]
3 27.825289 [(27.825289, 3.981281)]
...
97 0.0027837753 [(0.0027837753, 1.0002983)]
98 0.0025234222 [(0.0025234222, 1.0002704)]
99 0.0022875469 [(0.0022875469, 1.0002451)]
100 0.0020739238 [(0.0020739238, 1.0002222)]
'''
