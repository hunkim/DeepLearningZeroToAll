# Lab 4 Multi-variable linear regression
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Make sure the shape and data are OK
print(x_data, "\nx_data shape:", x_data.shape)
print(y_data, "\ny_data shape:", y_data.shape)

# data output
'''
[[ 73.  80.  75.]
 [ 93.  88.  93.]
 ...
 [ 76.  83.  71.]
 [ 96.  93.  95.]] 
x_data shape: (25, 3)
[[152.]
 [185.]
 ...
 [149.]
 [192.]] 
y_data shape: (25, 1)
'''

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
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                                   feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost:", cost_val, "\nPrediction:\n", hy_val)

# train output
'''
0 Cost: 21027.0 
Prediction:
 [[22.048063 ]
 [21.619772 ]
 ...
 [31.36112  ]
 [24.986364 ]]
10 Cost: 95.976326 
Prediction:
 [[157.11063 ]
 [183.99283 ]
 ...
 [167.48862 ]
 [193.25117 ]]
 1990 Cost: 24.863274 
Prediction:
 [[154.4393  ]
 [185.5584  ]
 ...
 [158.27443 ]
 [192.79778 ]]
2000 Cost: 24.722485 
Prediction:
 [[154.42894 ]
 [185.5586  ]
 ...
 [158.24257 ]
 [192.79166 ]]
'''

# Ask my score
print("Your score will be ", sess.run(hypothesis, 
                                      feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ", sess.run(hypothesis,
                                        feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))

'''
Your score will be  [[ 181.73277283]]
Other scores will be  [[ 145.86265564]
 [ 187.23129272]]

'''
