# Lab 3 Minimizing Cost
# This is optional
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

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

# Minimize: Gradient Descent Magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# Get gradients
gvs = optimizer.compute_gradients(cost, [W])
# Optional: modify gradient if necessary
# gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
# Apply gradients
apply_gradients = optimizer.apply_gradients(gvs)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(100):
    gradient_val, gvs_val, _ = sess.run([gradient, gvs, apply_gradients]) # Same as sess.run(train)
    print(step, gradient_val, gvs_val)

'''
0 37.333332 [(37.333336, 5.0)]
1 33.84889 [(33.84889, 4.6266665)]
2 30.689657 [(30.689657, 4.2881775)]
3 27.825289 [(27.825289, 3.981281)]
4 25.228264 [(25.228264, 3.7030282)]
...
96 0.0030694802 [(0.0030694804, 1.0003289)]
97 0.0027837753 [(0.0027837753, 1.0002983)]
98 0.0025234222 [(0.0025234222, 1.0002704)]
99 0.0022875469 [(0.0022875469, 1.0002451)]
'''
