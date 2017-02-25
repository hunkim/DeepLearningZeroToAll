from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_data = [[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

model = Sequential()
model.add(Dense(output_dim=1, input_dim=2))

model.compile(loss='mse', optimizer='sgd')
model.fit(x_data, y_data, nb_epoch=1000)

y_predict = model.predict(np.array([[95.,100.,80]]))
print(y_predict)
