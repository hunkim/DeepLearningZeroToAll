from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

x_data = np.array([[1, 2, 1, 1],
                   [2, 1, 3, 2],
                   [3, 1, 3, 4],
                   [4, 1, 5, 5],
                   [1, 7, 5, 5],
                   [1, 2, 5, 6],
                   [1, 6, 6, 6],
                   [1, 7, 7, 7]],
                  dtype=np.float32)
y_data = np.array([[0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [1, 0, 0],
                   [1, 0, 0]],
                  dtype=np.float32)

nb_classes = 3

model = Sequential()
model.add(Dense(3, input_shape=(4,)))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(x_data, y_data, epochs=1000)

print(model.predict_classes(np.array([[1, 2, 1, 1]])))
print(model.predict_classes(np.array([[1, 2, 5, 6]])))
