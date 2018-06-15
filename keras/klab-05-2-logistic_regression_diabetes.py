from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:,:-1]
y_data = xy[:,-1]

model = Sequential()
model.add(Dense(1, input_dim=8, activation='sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.summary()
model.fit(x_data, y_data, epochs=2000)

h = model.predict(x_train)
c = np.float32(h > 0.5)
a = np.mean(np.equal(c, y_train))
print("\nHypothesis: \n", h, "\nCorrect (Y): \n", c, "\nAccuracy: ", a)
