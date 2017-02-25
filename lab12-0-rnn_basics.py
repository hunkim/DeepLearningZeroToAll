# http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
# http://learningtensorflow.com/index.html
# http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint
pp = pprint.PrettyPrinter(indent=4)
tf.reset_default_graph()
sess = tf.InteractiveSession()


# One cell RNN input_dim (3) -> output_dim (5)
hidden_size = 5
cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
print(cell.output_size, cell.state_size)

x_data = np.array([[[1, 2, 3]]], dtype=np.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval())
sess.close()

states = 0

tf.reset_default_graph()
sess = tf.InteractiveSession()

# One cell RNN input_dim (3) -> output_dim (5). sequence: 2
hidden_size = 5
cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
x_data = np.array([[[1, 2, 3],
                    [4, 5, 6]]], dtype=np.float32)
outputs, states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval())
sess.close()


tf.reset_default_graph()
sess = tf.InteractiveSession()
# One cell RNN input_dim (3) -> output_dim (5). sequence: 2
cell = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True)
x_data = np.array([[[1, 2, 3],
                    [4, 5, 6]],

                   [[7, 8, 9],
                    [10, 11, 12]],

                   [[13, 14, 15],
                    [16, 17, 18]], ], dtype=np.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, x_data, sequence_length=[1,2,1], dtype=tf.float32)
sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval())
sess.close()


tf.reset_default_graph()
sess = tf.InteractiveSession()
# One cell RNN input_dim (3) -> output_dim (5). sequence: 2

batch_size = 3
cell = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)

x_data = np.array([[[1, 2, 3],
                    [4, 5, 6]],

                   [[7, 8, 9],
                    [10, 11, 12]],

                   [[13, 14, 15],
                    [16, 17, 18]], ], dtype=np.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data,
                                     initial_state=initial_state, dtype=tf.float32)
sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval())
sess.close()


tf.reset_default_graph()
sess = tf.InteractiveSession()
# Create input data
x_data = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
pp.pprint(x_data)  # batch, sequence_length, input size

# Make rnn
cell = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True)
cell = rnn.MultiRNNCell([cell] * 3, state_is_tuple=True)

# rnn in/out
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
print("dynamic rnn: ", outputs)
sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval())  # batch size, unrolling (time), hidden_size

cell = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32,
                                     sequence_length=[1, 2])
print("dynamic rnn: ", outputs)
sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval())  # batch size, unrolling (time), hidden_size
sess.close()


tf.reset_default_graph()
sess = tf.InteractiveSession()
# bi-directional rnn
cell_fw = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True)
cell_bw = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True)

outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x_data,
                                                  sequence_length=[2, 3],
                                                  dtype=tf.float32)

sess.run(tf.global_variables_initializer())
pp.pprint(sess.run(outputs))
pp.pprint(sess.run(states))

# Broadcasting based softmax
softmax_w = np.arange(12, dtype=np.float32).reshape(4, 3)
outputs = x_data * softmax_w
pp.pprint(softmax_w)
pp.pprint(outputs)
outputs = x_data * softmax_w + [1, 2, 3]
pp.pprint(outputs)

# [batch_size, sequence_length, emb_dim ]
prediction1 = tf.constant([[[0, 1], [0, 1], [0, 1]]], dtype=tf.float32)
prediction2 = tf.constant([[[1, 0], [1, 0], [1, 0]]], dtype=tf.float32)
prediction3 = tf.constant([[[0, 1], [1, 0], [0, 1]]], dtype=tf.float32)

# [batch_size, sequence_length]
y_data = tf.constant([[1, 1, 1]])

# [batch_size * sequence_length]
weights = tf.constant([[1, 1, 1]], dtype=tf.float32)

sequence_loss1 = tf.contrib.seq2seq.sequence_loss(prediction1, y_data, weights)
sequence_loss2 = tf.contrib.seq2seq.sequence_loss(prediction2, y_data, weights)
sequence_loss3 = tf.contrib.seq2seq.sequence_loss(prediction3, y_data, weights)

sess.run(tf.global_variables_initializer())
print("Loss1: ", sequence_loss1.eval(),
      "Loss2: ", sequence_loss2.eval(),
      "Loss3: ", sequence_loss3.eval())
