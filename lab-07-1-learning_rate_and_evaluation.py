# Lab 7 Learning rate and Evaluation
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

x_data = [[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5],
          [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
          [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]


# Evaluation our model using this test dataset
x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])

W = tf.Variable(tf.zeros([3, 3]))

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W))

# Cross entropy cost
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# Try to change learning_rate to small numbers
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Correct prediction Test model
prediction = tf.arg_max(hypothesis, 1)
is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={
                  X: x_data, Y: y_data}), sess.run(W))

    # predict
    print("Prediction:", sess.run(prediction, feed_dict={X: x_test}))
    # Calculate the accuracy
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))

'''
when lr = 10.
0 nan [[-0.83333319  0.4166666   0.41666645]
 [ 1.66666687  2.91666746 -4.58333397]
 [ 1.66666627  4.16666698 -5.83333397]]
200 nan [[ nan  nan  nan]
 [ nan  nan  nan]
 [ nan  nan  nan]]

...

2000 nan [[ nan  nan  nan]
 [ nan  nan  nan]
 [ nan  nan  nan]]
Prediction: [0 0 0]
Accuracy:  0.0

-------------------------------------------------
When lr = 0.1
0 1.0678 [[-0.00833333  0.00416667  0.00416666]
 [ 0.01666667  0.02916667 -0.04583334]
 [ 0.01666666  0.04166667 -0.05833334]]
200 0.699681 [[-1.5737735  -0.36410642  1.93788099]
 [ 0.0967759  -0.09235884 -0.00441682]
 [ 0.24915566  0.23823613 -0.48739177]]

...

1800 0.376595 [[-6.54242659  0.13831529  6.40411282]
 [ 0.13741526  0.02592302 -0.16333508]
 [ 1.16357517  0.11571171 -1.27928567]]
2000 0.361591 [[-6.95501471  0.21433842  6.7406764 ]
 [ 0.13668649  0.02956286 -0.16624542]
 [ 1.2420913   0.1036661  -1.3457557 ]]
Prediction: [2 2 2]
Accuracy:  1.0
'''
