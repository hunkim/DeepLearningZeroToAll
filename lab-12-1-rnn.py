# Lab 12 RNN
#WIP
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint
pp = pprint.PrettyPrinter(indent=4)

sess = tf.InteractiveSession()

h = [1,0,0,0]
e = [0,1,0,0]
l = [0,0,1,0]
o = [0,0,0,1]

hidden_size = 5
time_step_size = 4  # 'hell' -> 'ello'
batch_size = 1

num_classes = 4
sequence_length = 4
input_dim = 5

x_data = np.array([[h, e, l, l, o]], dtype=np.float32)
print(x_data)

cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, initial_state=initial_state, dtype=tf.float32)

prediction = tf.constant([[h, e, l, l, o]], dtype=tf.float32)
y_data = tf.Variable([[2, 3, 3, 4]])
print(y_data)
weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(prediction, y_data, weights)
cost = tf.reduce_sum(sequence_loss)/batch_size
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

sess.run(tf.global_variables_initializer())
pp.pprint(outputs)
print(sequence_loss.eval())