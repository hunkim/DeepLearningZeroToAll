
# http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# Video: https://www.youtube.com/watch?v=ftMq5ps503w
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from sklearn.preprocessing import MinMaxScaler
import os

# brew install graphviz
# pip3 install graphviz
# pip3 install pydot-ng
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt

timesteps = seq_length = 7
data_dim = 5

# Open,High,Low,Close,Volume
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)

# very important. It does not work without it.
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))
x = X_scaler.fit_transform(xy)
y = Y_scaler.fit_transform(xy[:, [-1]])  # Close as label

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# split to train and testing
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])

model = Sequential()
model.add(LSTM(5, input_shape=(timesteps, data_dim), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(10, return_sequences=False))
model.add(Dense(1))
model.add(Activation('linear'))

model.summary()

# Store model graph in png
# (Error occurs on in python interactive shell)
plot_model(model, to_file=os.path.basename(__file__) + '.png', show_shapes=True)

model.compile(loss='mean_squared_error', optimizer='adam')

print(trainX.shape, trainY.shape)
model.fit(trainX, trainY, epochs=20)

# make predictions
testPredict = model.predict(testX)

# inverse values
testPredict = Y_scaler.inverse_transform(testPredict)
testY = Y_scaler.inverse_transform(testY)

print(testPredict)
plt.plot(testY)
plt.plot(testPredict)
plt.show()