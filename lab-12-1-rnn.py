# Lab 12 RNN
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint
pp = pprint.PrettyPrinter(indent=4)

sess = tf.InteractiveSession()

char_rdic = ['h', 'e', 'l', 'o']  # id -> char
char_dic = {w: i for i, w in enumerate(char_rdic)}  # char -> id

sample = [char_dic[c] for c in "hello"]

# Configuration
char_vocab_size = len(char_dic)
rnn_size = char_vocab_size  # one hot encoding
time_step_size = 4  # 'hell' -> 'ello'
batch_size = 1

hidden_size = 5
cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
x_data = np.array([[[1, 0, 0,0], [0, 1, 0,0], [0, 0, 1,0],
                   [0, 0, 0, 1]]], dtype=np.float32)
outputs, states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

#logits: list of 2D Tensors of shape [batch_size x num_decoder_symbols]
#targets: list of 1D batch-sized int32 tensors of the same length as logits
#weights: list of 1D batch-sized float-Tensors of the same length as logits
logits = tf.reshape(tf.concat(values=outputs, axis=1), [-1, rnn_size])
targets = tf.reshape(sample[1:], [-1])
weights = tf.ones([time_step_size * batch_size])

print(len(outputs.get_shape()))
print(len(logits.get_shape()))

#ValueError: Logits must be a [batch_size x sequence_length x logits] tensor
loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=targets, weights=weights)

cost = tf.reduce_sum(loss)/batch_size

train_op = tf.train.GradientDescentOptimizer(
    learning_rate=.1).minimize(cost)


# Initialize variables
init = tf.global_variables_initializer()

# Launch graph
with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
    	sess.run(train_op)
    	result = sess.run(tf.argmax(logits, 1))
    	print (result, [char_rdic[t] for t in result])

