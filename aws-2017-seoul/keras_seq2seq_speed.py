# https://gist.github.com/rouseguy/1122811f2375064d009dac797d59bae9
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, LSTM
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import os


digit = "0123456789"
alpha = "abcdefghij"

char_set = list(set(digit + alpha))  # id -> char
char_dic = {w: i for i, w in enumerate(char_set)}

data_dim = len(char_set)  # one hot encoding size
seq_length = time_steps = 7
num_classes = len(char_set)

# Build training date set
dataX = []
dataY = []

for i in range(1000):
    rand_pick = np.random.choice(10, 7)
    x = [char_dic[digit[c]] for c in rand_pick]
    y = [char_dic[alpha[c]] for c in rand_pick]
    dataX.append(x)
    dataY.append(y)

# One-hot encoding
dataX = np_utils.to_categorical(dataX, num_classes=num_classes)
# reshape X to be [samples, time steps, features]
dataX = np.reshape(dataX, (-1, seq_length, data_dim))

# One-hot encoding
dataY = np_utils.to_categorical(dataY, num_classes=num_classes)
# time steps
dataY = np.reshape(dataY, (-1, seq_length, data_dim))


print('Build model...')
TensorBoard(log_dir='./logs', histogram_freq=1,
            write_graph=True, write_images=False)

model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add(LSTM(32, input_shape=(time_steps, data_dim), return_sequences=False))

# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(time_steps))
# The decoder RNN could be multiple layers stacked or a single layer

model.add(LSTM(32, return_sequences=True))

# For each of step of the output sequence, decide which character should
# be chosen
model.add(TimeDistributed(Dense(data_dim)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam')
begin = time.time()
model.fit(dataX, dataY, epochs=100, verbose=2)
end = time.time()
print("Total Time Spent: %gs" %(end - begin))
