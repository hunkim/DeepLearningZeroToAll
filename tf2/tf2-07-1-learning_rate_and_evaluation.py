# Lab 7 Learning rate and Evaluation
import tensorflow as tf

x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# Evaluation our model using this test dataset
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

# try different learning_rate
# learning_rate = 65535  # ? it works too hahaha
learning_rate = 0.1
# learning_rate = 1e-10  # small learning rate won't work either

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=3, input_dim=3, activation='softmax'))
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=learning_rate), metrics=['accuracy'])

tf.model.fit(x_data, y_data, epochs=1000)

# predict
print("Prediction: ", tf.model.predict_classes(x_test))

# Calculate the accuracy
print("Accuracy: ", tf.model.evaluate(x_test, y_test)[1])
