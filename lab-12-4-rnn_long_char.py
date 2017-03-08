# Lab 12 RNN
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

data_dim = hidden_size = len(char_set) #25
seq_length = timesteps = 10
num_classes = len(char_set) #25

dataX = []
dataY = []

for i in range(0, len(sentence) - seq_length):
    x_str = sentence[i:i + seq_length]
    y_str = sentence[i + 1: i + seq_length + 1]

    x = [char_dic[c] for c in x_str]  # char to index
    y = [char_dic[c] for c in y_str]  # char to index

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX) #170

X = tf.placeholder(tf.int32, [None, seq_length])
Y = tf.placeholder(tf.int32, [None, seq_length])

# One-hot encoding
x_one_hot = tf.one_hot(X, num_classes)
print(x_one_hot)

# Make lstm with rnn_hidden_size (each unit input vector size)
lstm = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
lstm = rnn.MultiRNNCell([lstm] * 2, state_is_tuple=True)

# outputs: unrolling size x hidden size, state = hidden size
outputs, _states = tf.nn.dynamic_rnn(lstm, x_one_hot, dtype=tf.float32)

# (optional) softmax layer
x_for_softmax = tf.reshape(outputs, [-1, hidden_size])
softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b", [num_classes])
outputs = tf.matmul(x_for_softmax, softmax_w) + softmax_b

outputs = tf.reshape(outputs, [batch_size, seq_length, num_classes])
weights = tf.ones([batch_size, seq_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(outputs, Y, weights)
mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(mean_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    _, l, results = sess.run([train_op, mean_loss, outputs], feed_dict={X: dataX, Y:dataY})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        print(i, j, ''.join([char_set[t] for t in index]), l)

# Let's print the last char of each result
results = sess.run(outputs, feed_dict={X: dataX})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    print(char_set[index[-1]], end='')