from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_data = np.array([[0., 5], [2., 4.], [4., 3.], [6., 3.], [8., 1.]])
y_data = np.array([1, 2, 3, 4, 5])

print(x_data.shape)

model = Sequential()
model.add(Dense(output_dim=1, input_dim=2))

model.compile(loss='mse', optimizer='sgd')
model.fit(x_data, y_data)

y_predict = model.predict(x_data)
print(y_predict)
