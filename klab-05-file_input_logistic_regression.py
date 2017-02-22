from keras.models import Sequential
from keras.layers import Dense
import numpy as np

xy = np.loadtxt('data.csv', delimiter=",")
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print("x_data", x_data)
print("y_data", y_data)


model = Sequential()
model.add(Dense(input_dim=3, output_dim=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd')
model.fit(x_data, y_data, nb_epoch=1000)

y_predict = model.predict_classes(np.array([[0, 2, 1]]))
print(y_predict)

y_predict = model.predict_classes(np.array([[2, 9, 9]]))
print(y_predict)
