#! /usr/bin/env python
# -*- coding: utf-8 -*-

from caffe2.python import workspace, model_helper
import numpy as np

x = np.random.rand(4, 3, 2)
print(x)
print(x.shape)

workspace.FeedBlob("myx", x)

x2 = workspace.FetchBlob("myx")

print(x2)
