from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))

model.compile(loss='mse', optimizer='sgd')

# prints summary of the model to the terminal
model.summary()

model.fit(x_train, y_train, nb_epoch=1000)

y_predict = model.predict(np.array([4]))
print(y_predict)
