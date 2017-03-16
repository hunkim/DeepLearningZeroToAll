# https://www.tensorflow.org/api_guides/python/array_ops
import tensorflow as tf
import numpy as np
import pprint
tf.set_random_seed(777)  # for reproducibility

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

t = tf.random_normal([3])
pp.pprint(sess.run(t))
'''
array([ 2.20866942, -0.73225045,  0.33533147], dtype=float32)
'''

t = tf.random_uniform([2])
pp.pprint(sess.run(t))
'''
array([ 0.08186948,  0.42999184], dtype=float32)
'''

t = tf.random_uniform([2, 3])
pp.pprint(sess.run(t))
'''
array([[ 0.43535876,  0.76933432,  0.65130949],
       [ 0.90863407,  0.06278825,  0.85073185]], dtype=float32)
'''

t = tf.reduce_mean([1, 2], axis=0)
pp.pprint(sess.run(t))
''' (int division)
1
'''

x = [[1., 2.],
     [3., 4.]]


t = tf.reduce_mean(x)
pp.pprint(sess.run(t))
'''
2.5
'''

t = tf.reduce_mean(x, axis=0)
pp.pprint(sess.run(t))
'''
array([ 2.,  3.], dtype=float32)
'''

t = tf.reduce_mean(x, axis=1)
pp.pprint(sess.run(t))
'''
array([ 1.5,  3.5], dtype=float32)
'''

t = tf.reduce_mean(x, axis=-1)
pp.pprint(sess.run(t))
'''
array([ 1.5,  3.5], dtype=float32)
'''

t = tf.reduce_sum(x)
pp.pprint(sess.run(t))
'''
10.0
'''

t = tf.reduce_sum(x, axis=0)
pp.pprint(sess.run(t))
'''
array([ 4.,  6.], dtype=float32)
'''

t = tf.reduce_sum(x, axis=-1)
pp.pprint(sess.run(t))
'''
array([ 3.,  7.], dtype=float32)
'''

t = tf.reduce_mean(tf.reduce_sum(x, axis=-1))
pp.pprint(sess.run(t))
'''
5.0
'''

x = [[0, 1, 2],
     [2, 1, 0]]
t = tf.argmax(x, axis=0)
pp.pprint(sess.run(t))
'''
array([1, 0, 0])
'''

t = tf.argmax(x, axis=1)
pp.pprint(sess.run(t))
'''
array([2, 0])
'''

t = tf.argmax(x, axis=-1)
pp.pprint(sess.run(t))
'''
array([2, 0])
'''

t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
pp.pprint(t.shape)
pp.pprint(t)
'''
(2, 2, 3)
array([[[ 0,  1,  2],
        [ 3,  4,  5]],

       [[ 6,  7,  8],
        [ 9, 10, 11]]])
'''


t = tf.reshape(t, shape=[-1, 3])
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t))
'''
(4, 3)
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])
'''

t = tf.reshape(t, shape=[-1, 1, 3])
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t))
'''
(4, 1, 3)
array([[[ 0,  1,  2]],

       [[ 3,  4,  5]],

       [[ 6,  7,  8]],

       [[ 9, 10, 11]]])
'''

t = tf.squeeze([[0], [1], [2]])
pp.pprint(sess.run(t))
'''
array([0, 1, 2], dtype=int32)
'''

t = tf.expand_dims([0, 1, 2], 1)
pp.pprint(sess.run(t))
'''
array([[0],
       [1],
       [2]], dtype=int32)
'''

t = tf.one_hot([[0], [1], [2], [0]], depth=3)
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t))
'''
(4, 1, 3)
array([[[ 1.,  0.,  0.]],

       [[ 0.,  1.,  0.]],

       [[ 0.,  0.,  1.]],

       [[ 1.,  0.,  0.]]], dtype=float32)
'''

t = tf.reshape(t, shape=[-1, 3])
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t))
'''
(4, 3)
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.],
       [ 1.,  0.,  0.]], dtype=float32)
'''


x = [1, 4]
y = [2, 5]
z = [3, 6]

# Pack along first dim.
t = tf.stack([x, y, z])
pp.pprint(sess.run(t))
'''
array([[1, 4],
       [2, 5],
       [3, 6]], dtype=int32)
'''

t = tf.stack([x, y, z], axis=1)
pp.pprint(sess.run(t))
'''
array([[1, 2, 3],
       [4, 5, 6]], dtype=int32)
'''

t = tf.cast([1.8, 2.2], tf.int32)
pp.pprint(sess.run(t))
'''
array([1, 2], dtype=int32)
'''

t = tf.cast([True, False, 1 == 1, 0 == 1], tf.int32)
pp.pprint(sess.run(t))
'''
array([1, 0, 1, 0], dtype=int32)
'''

for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)
'''
1 4
2 5
3 6
'''

for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)
'''
1 4 7
2 5 8
3 6 9
'''


t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
pp.pprint(t.shape)
pp.pprint(t)
'''
(2, 2, 3)
array([[[ 0,  1,  2],
        [ 3,  4,  5]],

       [[ 6,  7,  8],
        [ 9, 10, 11]]])
'''

t1 = tf.transpose(t, [1, 0, 2])
pp.pprint(sess.run(t1).shape)
pp.pprint(sess.run(t1))
'''
(2, 2, 3)
array([[[ 0,  1,  2],
        [ 6,  7,  8]],

       [[ 3,  4,  5],
        [ 9, 10, 11]]])
'''

t = tf.transpose(t1, [1, 0, 2])
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t))
'''
(2, 2, 3)
array([[[ 0,  1,  2],
        [ 3,  4,  5]],

       [[ 6,  7,  8],
        [ 9, 10, 11]]])
'''

t2 = tf.transpose(t, [1, 2, 0])
pp.pprint(sess.run(t2).shape)
pp.pprint(sess.run(t2))
'''
(2, 3, 2)
array([[[ 0,  6],
        [ 1,  7],
        [ 2,  8]],

       [[ 3,  9],
        [ 4, 10],
        [ 5, 11]]])
'''

t = tf.transpose(t2, [2, 0, 1])
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t))
'''
(2, 2, 3)
array([[[ 0,  1,  2],
        [ 3,  4,  5]],

       [[ 6,  7,  8],
        [ 9, 10, 11]]])
'''
