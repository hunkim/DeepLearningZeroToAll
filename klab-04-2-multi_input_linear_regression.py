from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_data = np.array([[1, 2, 1], [1, 3, 2], [1, 3, 4],
                   [1, 5, 5], [1, 7, 5], [1, 2, 5]])
y_data = np.array([0, 0, 0, 1, 1, 1])

print(x_data.shape)

model = Sequential()
model.add(Dense(input_dim=3, output_dim=1))

model.compile(loss='mse', optimizer='sgd')
model.fit(x_data, y_data)

y_predict = model.predict(x_data)
print(y_predict)
