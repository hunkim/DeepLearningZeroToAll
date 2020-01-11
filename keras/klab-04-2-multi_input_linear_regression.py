from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_data = np.array([[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]])
y_data = np.array([[152.],
          [185.],
          [180.],
          [196.],
          [142.]])

model = Sequential()
model.add(Dense(input_dim=3, units=1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_data, y_data, epochs=500)

y_predict = model.predict(np.array([[0, 2, 1]]))
print(y_predict)