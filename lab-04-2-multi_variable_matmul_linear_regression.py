# Lab 4 Multi-variable linear regression
import tensorflow as tf

x_data = [[1., 1.], [2., 2.], [3., 3.],
          [4., 4.], [5., 5.]]
y_data = [[1], [2], [3], [4], [5]]

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(x_data, W) + b

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch graph
sess = tf.Session()
# Initialize TensorFlow variables
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost),
              sess.run(hypothesis), sess.run(W), sess.run(b))
