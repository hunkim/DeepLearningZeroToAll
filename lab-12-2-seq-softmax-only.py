import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility

sample = " if you want you"
char_set = list(set(sample))  # id -> char
char_dic = {w: i for i, w in enumerate(char_set)}

# hyper parameters
dic_size = len(char_dic)  # RNN input size (one hot size)
rnn_hidden_size = len(char_dic)  # RNN output size
num_classes = len(char_dic)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)

sample_idx = [char_dic[c] for c in sample]  # char to index
x_data = sample_idx[:-1]  # X data sample (0~n-1)
y_data = sample_idx[1:]   # Y label sample (1~n)

x_data = tf.one_hot(x_data, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0

# flatten the data (ignore batches for now). No effect if the batch size is 1
x_data = tf.reshape(x_data, [-1, rnn_hidden_size])

# softmax layer (rnn_hidden_size -> num_classes)
softmax_w = tf.get_variable("softmax_w", [rnn_hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b", [num_classes])
outputs = tf.matmul(x_data,  softmax_w) + softmax_b

# expend the data (revive the batches)
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
y_data = tf.reshape(y_data, [batch_size, sequence_length])
weights = tf.ones([batch_size, sequence_length])

# Compute sequence loss
sequence_loss = tf.contrib.seq2seq.sequence_loss(outputs, y_data, weights)
mean_loss = tf.reduce_mean(sequence_loss)  # mean all sequence loss
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(mean_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, l = sess.run([train_op, mean_loss])
    results = sess.run(outputs)
    for result in results:  # n-batch outputs
        index = np.argmax(result, axis=1)
        print(''.join([char_set[t] for t in index]), l)
