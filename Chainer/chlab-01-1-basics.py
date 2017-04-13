#!/usr/bin/env python
# Lab 1-1 Basics

import numpy as np
import chainer


# Create Variable object.
a = chainer.Variable(np.array([1], dtype=np.float32))
b = chainer.Variable(np.array([2], dtype=np.float32))

# Variable object has basic arithmetic operators.
y = a * b

# Now y is a Variable object, with attribute "data".
# http://docs.chainer.org/en/latest/reference/core/variable.html?highlight=Variable
print("{} should equal [ 2.]".format(y.data))

a = chainer.Variable(np.array([3], dtype=np.float32))
b = chainer.Variable(np.array([3], dtype=np.float32))

y = a * b
print("{} should equal [ 9.]".format(y.data))

"""
Expected output.
---
[ 2.] should equal [ 2.]
[ 9.] should equal [ 9.]
"""
