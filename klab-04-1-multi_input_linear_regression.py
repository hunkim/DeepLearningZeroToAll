from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_data = [[1., 0.], [0., 2.], [3., 0.], [0., 4.], [5., 0.]]
y_data = [[1], [2], [3], [4], [5]]

model = Sequential()
model.add(Dense(output_dim=1, input_dim=2))

model.compile(loss='mse', optimizer='sgd')
model.fit(x_data, y_data, nb_epoch=1000)

y_predict = model.predict(np.array([[2.1, 4.2]]))
print(y_predict)
