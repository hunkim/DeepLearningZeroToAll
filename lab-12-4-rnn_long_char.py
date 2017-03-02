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
Y = tf.placeholder(tf.int64, [None, seq_length])

# One-hot encoding
x_one_hot = tf.one_hot(X, num_classes)
dropout = 0.5

cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_classes, state_is_tuple=True)
# cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
cell = tf.contrib.rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)

outputs, _states = tf.nn.dynamic_rnn(cell, x_one_hot, dtype=tf.float32)

weights = tf.ones([batch_size, seq_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
predict = tf.argmax(outputs, axis=-1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for x in range(2000):
        l, _, a = sess.run([loss, train, accuracy], feed_dict={X: dataX, Y: dataY})
        result = sess.run(predict, feed_dict={X: dataX})
        x_index = dataX[0]
        x_str = [char_set[j] for j in x_index]

        index = result[0]
        result = [char_set[j] for j in index]
        print(''.join(x_str), ' -> ', ''.join(result))



        print(x, "loss: ", l, "accuracy: ", a)

    for i, prediction in enumerate(result):
        x_index = dataX[i]
        x_str = [char_set[j] for j in x_index]

        index = prediction
        result = [char_set[j] for j in index]

        print(''.join(x_str), ' -> ', ''.join(result))
