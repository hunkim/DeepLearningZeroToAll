# Lab 12 RNN
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

data_dim = hidden_size = len(char_set)
seq_length = timesteps = 10
num_classes = len(char_set)

dataX = []
dataY = []

for i in range(0, len(sentence) - seq_length):
    x_str = sentence[i:i + seq_length]
    y_str = sentence[i + 1: i + seq_length + 1]

    x = [char_dic[c] for c in x_str]  # char to index
    y = [char_dic[c] for c in y_str]  # char to index

    dataX.append(x)
    dataY.append(y)

dataX = np.asarray(dataX, dtype=np.float32)
datay = np.asarray(dataY, dtype=np.float32)

batch_size = len(dataX) #170
print(dataX)

X = tf.placeholder(tf.int32, [None, seq_length])
Y = tf.placeholder(tf.int32, [None, seq_length])

# One-hot encoding
x_one_hot = tf.one_hot(X, num_classes)
# One-hot encoding
y_one_hot = tf.one_hot(Y, num_classes)

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
cell = tf.contrib.rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)

outputs, _states = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

weights = tf.ones([batch_size, seq_length])
loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

predict = tf.argmax(outputs, 1)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(2000):
		l, _ = sess.run([loss, train], feed_dict={X: dataX, Y: dataY})
		result = sess.run(predict, feed_dict={X: dataX})
		if i == 1999:
			print(i, "loss:", l, "prediction: ", result, "true Y: ", dataY)
		print(i, "loss: ", l)