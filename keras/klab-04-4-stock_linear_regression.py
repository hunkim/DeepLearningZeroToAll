from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')

# very important. It does not work without it.
scaler = MinMaxScaler(feature_range=(0, 1))
xy = scaler.fit_transform(xy)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# before deletion
print(x_data[0], y_data[0])
print(x_data[1], y_data[1])

# predict tomorrow
x_data = np.delete(x_data, -1, 0)
y_data = np.delete(y_data, 0)

print("== Predict tomorrow")
print(x_data[0], "->", y_data[0])

model = Sequential()
model.add(Dense(input_dim=4, units=1))

model.compile(loss='mse', optimizer='sgd', metrics=['mse'])
model.fit(x_data, y_data, epochs=100)

test = x_data[10].reshape(-1, 4)
print("y=", y_data[10], "prediction=", model.predict(test))

test = x_data[30].reshape(-1, 4)
print("y=", y_data[30], "prediction=", model.predict(test))

# ---------------------------
# Test
# split to train and testing

train_size = int(len(x_data) * 0.7)
test_size = len(x_data) - train_size
x_train, x_test = x_data[0:train_size], x_data[train_size:len(x_data)]
y_train, y_test = y_data[0:train_size], y_data[train_size:len(y_data)]

model = Sequential()
model.add(Dense(input_dim=4, units=1))
model.compile(loss='mse', optimizer='sgd', metrics=['mse'])

# Train a model
model.fit(x_train, y_train, epochs=200)

# evaluate
results = model.evaluate(x_test, y_test, verbose=1)
print(results)

predictions = model.predict(x_test)

plt.plot(y_test)
plt.plot(predictions)
plt.show()
