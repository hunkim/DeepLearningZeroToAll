# Lab 4 Multi-variable linear regression
import tensorflow as tf

x1_data = [1, 0, 3, 0, 5]
x2_data = [0, 2, 0, 4, 0]
y_data = [1, 2, 3, 4, 5]

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1_data * w1 + x2_data * w2 + b
print(hypothesis)

# Cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch graph
sess = tf.Session()
# Initialize TensorFlow variables
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print("%s w1: %s w2: %s b: %s cost: %s" %
              (step, sess.run(w1), sess.run(w2), sess.run(b), sess.run(cost)))
