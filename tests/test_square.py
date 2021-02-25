# https://www.tensorflow.org/api_guides/python/test

import tensorflow as tf
import numpy as np


class SquareTest(tf.test.TestCase):

    def testSquare(self):
        with self.test_session():
            x = tf.square([2, 3])
            self.assertAllEqual(x, [4, 9])

    def testBroadcast(self):
        with self.test_session():
            hypothesis = np.array([[1], [2], [3]])
            y = np.array([1, 2, 3])
            print(hypothesis - y)
            cost = tf.reduce_mean(tf.square(hypothesis - y))
            self.assertNotEqual(cost, 0)

            y = y.reshape(-1, 1)
            print(y, hypothesis - y)
            cost = tf.reduce_mean(tf.square(hypothesis - y))
            self.assertAllEqual(cost, 0)

    def testNormalize(self):
        with self.test_session():
            values = np.array([[10, 20], [1000, -100]], dtype=np.float32)
            norm_values = tf.nn.l2_normalize(values, axis=1)
            print(norm_values)

if __name__ == '__main__':
    tf.test.main()
