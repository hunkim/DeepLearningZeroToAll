from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout


batch_size = 128
num_classes = 10
epochs = 12


# ==============================================================================
# prepare data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# ==============================================================================
# build model
# (model code from http://iostream.tistory.com/111)
model = Sequential()

model.add(Dense(256, input_dim=784,
                kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(256, kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(256, kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(256, kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.2)


# ==============================================================================
# predict
score = model.evaluate(X_test, y_test, batch_size=batch_size)
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])
'''
Test loss: 0.0742975851574
Test accuracy: 0.9811
'''
