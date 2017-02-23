from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler

xy = np.loadtxt('stock_daily.csv', delimiter=',')

# very important. It does not work without it.
scaler = MinMaxScaler(feature_range=(0, 1))
xy = scaler.fit_transform(xy)

x_data = xy[:,0:-1]
y_data = xy[:, [-1]]

# before deletion
print(x_data[0], y_data[0])
print(x_data[1], y_data[1])

# predict tomorrow
x_data = np.delete(x_data, -1,0)
y_data = np.delete(y_data, 0)

print("== Predict tomorrow")
print(x_data[0], "->", y_data[0])

model = Sequential()
model.add(Dense(input_dim=4, output_dim=1))

model.compile(loss='mse', optimizer='sgd')
model.fit(x_data, y_data, nb_epoch=100)

test = x_data[10].reshape(-1,4)
print("y=", y_data[10], "prediction=", model.predict(test))
