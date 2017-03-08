# Lab 12 RNN
# WIP
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint
pp = pprint.PrettyPrinter(indent=4)

h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

hidden_size = 4
time_step_size = 4  # 'hell' -> 'ello'
batch_size = 1

num_classes = 4
sequence_length = 4
input_dim = 4

x_data = np.array([[h, e, l, l]], dtype=np.float32)

cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, x_data, initial_state=initial_state, dtype=tf.float32)

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
y_data = tf.Variable([[1, 2, 2, 3]], name="y_data")
weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=y_data, weights=weights)

train = tf.train.GradientDescentOptimizer(
    learning_rate=0.1).minimize(tf.reduce_mean(sequence_loss))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        sess.run(train)
        result = sess.run(tf.arg_max(outputs, 1))
        print(result)
