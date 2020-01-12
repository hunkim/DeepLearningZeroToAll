# Lab 2 Linear Regression
import tensorflow as tf

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Try to find values for W and b to compute y_data = x_data * W + b
# We know that W should be 1 and b should be 0
# But let TensorFlow figure it out
W = tf.Variable(tf.random.normal([1], seed=777), name="weight")
b = tf.Variable(tf.random.normal([1], seed=777), name="bias")

# Our hypothesis XW+b
hypothesis = lambda: x_train * W + b

# cost/loss function
cost_val = 0
def cost():
    global cost_val
    cost_val = tf.reduce_mean(tf.square(hypothesis() - y_train))
    return cost_val

# optimizer
train = lambda: tf.optimizers.SGD(learning_rate=0.01).minimize(cost, [W,b])

# Fit the line
for step in range(2001):
    train()
    if step % 20 == 0:
        print(step, cost_val.numpy(), W.numpy(), b.numpy())

# Learns best fit W:[ 1.],  b:[ 0.]
"""
0 2.82329 [ 2.12867713] [-0.85235667]
20 0.190351 [ 1.53392804] [-1.05059612]
40 0.151357 [ 1.45725465] [-1.02391243]
...
1960 1.46397e-05 [ 1.004444] [-0.01010205]
1980 1.32962e-05 [ 1.00423515] [-0.00962736]
2000 1.20761e-05 [ 1.00403607] [-0.00917497]
"""
