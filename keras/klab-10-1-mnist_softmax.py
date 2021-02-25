from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras import initializers
from keras.utils import np_utils
from keras import backend as K

from keras.datasets import mnist

# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# parameters
learning_rate = 0.001
batch_size = 100
training_epochs = 15
nb_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], img_rows * img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_rows * img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows * img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_rows * img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(units=nb_classes, input_dim=img_rows * img_cols,
                kernel_initializer=initializers.random_normal(stddev=0.01),
                use_bias=True))
model.add(Activation('softmax'))

model.summary()

adam = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, epochs=training_epochs)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

'''
Test score: 0.265781824633
Test accuracy: 0.9268
'''
