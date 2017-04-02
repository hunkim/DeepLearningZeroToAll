# https://github.com/fchollet/keras/tree/master/examples
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
import numpy as np

# Predicting animal type based on various features
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]] - 1
print(x_data.shape, y_data.shape)

nb_classes = 7
y_one_hot = np_utils.to_categorical(y_data, nb_classes)

model = Sequential()
model.add(Dense(nb_classes, input_shape=(16,)))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(x_data, y_one_hot, epochs=1000)

# Let's see if we can predict
pred = model.predict_classes(x_data)
for p, y in zip(pred, y_data):
    print("prediction: ", p, " true Y: ", y)
