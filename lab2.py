#Lab 2 Linear Regression

import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

#Try to find values for W and b to compute y_data = W * x_data + b
#We know that W should be 1 and b should be 0
#But let's use Tensorflow to figure it out
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

#Our hypothesis
hypothesis = W * x_data + b

#Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

#Minimize
a = tf.Variable(0.1) #Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

#Initialize all variables before we start
init = tf.global_variables_initializer()

#Lauch graph
sess = tf.Session()
sess.run(init)

#Fit the line
for step in range(2001):
	sess.run(train)
	if step % 20 == 0:
		print(step, sess.run(cost), sess.run(W), sess.run(b))

#Learns best fit W:[ 1.],  b:[  5.83677604e-08]