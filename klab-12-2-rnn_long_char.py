import numpy as np
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Activation, LSTM
from keras.utils import np_utils

# hello
sentence = "If you want to build a ship, don’t drum up people together to collect wood and don’t assign them tasks and work, but rather teach them to long for the endless immensity of the sea."

char_set = list(set(sentence))  # id -> char ['i', 'l', 'e', 'o', 'h']
char_dic = {w:i for i, w in enumerate(char_set)}

data_dim = len(char_set)
seq_length = timesteps = 10
nb_classes = len(char_set)

dataX = []
dataY = []
for i in range(0, len(sentence) - seq_length):
    x_str = sentence[i:i + seq_length]
    y_str = sentence[i+1: i + seq_length+1]
    print(x_str, '->', y_str)

    x = [char_dic[c] for c in x_str] # char to index
    y = [char_dic[c] for c in y_str] # char to index

    dataX.append(x)
    dataY.append(y)

# One-hot encoding
x = np_utils.to_categorical(dataX, nb_classes=nb_classes)
# reshape X to be [samples, time steps, features]
x = np.reshape(x, (-1, seq_length, data_dim))
print(x.shape)

# One-hot encoding
y = np_utils.to_categorical(dataY, nb_classes=nb_classes)
# time steps
y = np.reshape(y, (-1, seq_length, data_dim))
print(y.shape)

model = Sequential()
model.add(LSTM(nb_classes, input_shape=(timesteps, data_dim), return_sequences=True))
model.add(LSTM(nb_classes, return_sequences=True))
model.add(TimeDistributed(Dense(nb_classes)))

model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
model.fit(x, y, nb_epoch=1000)

predictions = model.predict(x, verbose=0)
for i, prediction in enumerate(predictions):
    # print(prediction)
    x_index = np.argmax(x[i], axis=1)
    x_str = [char_set[j] for j in x_index]
    print(x_index, ''.join(x_str))

    index = np.argmax(prediction, axis=1)
    result = [char_set[j] for j in index]
    print(index, ''.join(result))