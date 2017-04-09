from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

model = Sequential()
model.add(Dense(input_dim=3, units=1))

model.compile(loss='mse', optimizer='rmsprop')
model.fit(x_data, y_data)

y_predict = model.predict(np.array([[0, 2, 1]]))
print(y_predict)
