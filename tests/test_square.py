# https://www.tensorflow.org/api_guides/python/test

import tensorflow as tf
import numpy as np


class SquareTest(tf.test.TestCase):

    def testSquare(self):
        with self.test_session():
            x = tf.square([2, 3])
            self.assertAllEqual(x.eval(), [4, 9])

    def testBroadcast(self):
        with self.test_session():
            hypothesis = np.array([[1], [2], [3]])
            y = np.array([1, 2, 3])
            print("broadcast", hypothesis - y)
            cost = tf.reduce_mean(tf.square(hypothesis - y))
            self.assertNotEqual(cost.eval(), 0)

            y = y.reshape(-1, 1)
            print("no broadcast", y, hypothesis - y)
            cost = tf.reduce_mean(tf.square(hypothesis - y))
            self.assertAllEqual(cost.eval(), 0)

    def testSquaredDifference(self):
        with self.test_session():
            hypothesis = np.array([[1], [2], [3]])
            y = np.array([1, 2, 3])
            diff = tf.squared_difference(hypothesis, y)
            print("squared difference", diff.eval())
            cost = tf.reduce_mean(diff)
            self.assertNotEqual(cost.eval(), 0)

if __name__ == '__main__':
    tf.test.main()
