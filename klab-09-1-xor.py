from keras.models import Sequential
from keras.layers import Dense

x_data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y_data = [[0.], [1.], [1.], [0.]]

model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd',
              lr=0.1, metrics=['accuracy'])
model.summary()
model.fit(x_data, y_data, nb_epoch=50000)

score = model.evaluate(x_data, y_data, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
