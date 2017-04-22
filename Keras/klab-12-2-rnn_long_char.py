import numpy as np
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Activation, LSTM
from keras.utils import np_utils

import os

# brew install graphviz
# pip3 install graphviz
# pip3 install pydot-ng
from keras.utils.vis_utils import plot_model

# sample sentence
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))  # id -> char ['i', 'l', 'e', 'o', 'h', ...]
char_dic = {w: i for i, w in enumerate(char_set)}

data_dim = len(char_set)
seq_length = timesteps = 10
num_classes = len(char_set)

dataX = []
dataY = []
for i in range(0, len(sentence) - seq_length):
    x_str = sentence[i:i + seq_length]
    y_str = sentence[i + 1: i + seq_length + 1]
    print(x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]  # char to index
    y = [char_dic[c] for c in y_str]  # char to index

    dataX.append(x)
    dataY.append(y)

# One-hot encoding
dataX = np_utils.to_categorical(dataX, num_classes=num_classes)
# reshape X to be [samples, time steps, features]
dataX = np.reshape(dataX, (-1, seq_length, data_dim))
print(dataX.shape)

# One-hot encoding
dataY = np_utils.to_categorical(dataY, num_classes=num_classes)
# time steps
dataY = np.reshape(dataY, (-1, seq_length, data_dim))
print(dataY.shape)

model = Sequential()
model.add(LSTM(num_classes, input_shape=(
    timesteps, data_dim), return_sequences=True))
model.add(LSTM(num_classes, return_sequences=True))
model.add(TimeDistributed(Dense(num_classes)))

model.add(Activation('softmax'))
model.summary()

# Store model graph in png
# (Error occurs on in python interactive shell)
plot_model(model, to_file=os.path.basename(__file__) + '.png', show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])
model.fit(dataX, dataY, epochs=1000)

predictions = model.predict(dataX, verbose=0)
for i, prediction in enumerate(predictions):
    # print(prediction)
    x_index = np.argmax(dataX[i], axis=1)
    x_str = [char_set[j] for j in x_index]

    index = np.argmax(prediction, axis=1)
    result = [char_set[j] for j in index]

    print(''.join(x_str), ' -> ', ''.join(result))
