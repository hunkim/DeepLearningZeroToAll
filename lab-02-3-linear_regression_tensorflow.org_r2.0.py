# From https://www.tensorflow.org/get_started/get_started
import tensorflow as tf

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# Model parameters
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)

hypothesis = lambda: x * W + b

# cost/loss function
cost_val = 0
def cost():
    global cost_val
    cost_val = tf.reduce_sum(tf.square(hypothesis() - y))  # sum of the squares
    return cost_val

# optimizer
train = lambda: tf.optimizers.SGD(learning_rate=0.01).minimize(cost, [W,b])

x = x_train
y = y_train
for step in range(1000):
    train()

# evaluate training accuracy
print(f"W: {W.numpy()} b: {b.numpy()} cost: {cost_val.numpy()}")

"""
W: [-0.9999969] b: [0.9999908] cost: 5.699973826267524e-11
"""
