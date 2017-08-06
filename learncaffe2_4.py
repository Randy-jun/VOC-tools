#! /usr/bin/env python
# -*- coding: utf-8 -*-

from caffe2.python import workspace, model_helper
import numpy as np


x = np.random.rand(4, 3, 2)
y = np.random.rand(1, 3, 2)
z = np.random.rand(1, 2, 2)
print(x)
print(y)
print(z)

np.savez("result.npz", x, label = y, res = z)
loa = np.load("result.npz")
print(type(loa))
print(loa["arr_0"])
