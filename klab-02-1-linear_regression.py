from keras.models import Sequential
from keras.layers import Dense, Activation

x_data = [1, 2, 3]
y_data = [1, 2, 3]

model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))

model.compile(loss='mse', optimizer='sgd')
model.fit(x_data, x_data)

y_predict = model.predict(x_data)
print(y_predict)
