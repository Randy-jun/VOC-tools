#! /usr/bin/env python
# -*- coding: utf-8 -*-

from caffe2.python import workspace, model_helper
import numpy as np

data = np.random.rand(16, 100).astype(np.float32)

label = (np.random.rand(16) * 10).astype(np.int32)
# mini-batch is 16
workspace.FeedBlob("data", data)
workspace.FeedBlob("label", label)

m = model_helper.ModelHelper(name = "my_first_net")

weight = m.param_init_net.XavierFill([], 'fc_w', shape = [10, 100])
bias = m.param_init_net.ConstantFill([], 'fc_b', shape = [10, ])

fc_1 = m.net.FC(["data", "fc_w", "fc_b"], "fc1")
pred = m.net.Sigmoid(fc_1, "pred")
[softmax, loss] = m.net.SoftmaxWithLoss([pred, "label"], ["softmax", "loss"])

m.AddGradientOperators([loss])

print(str(m.net.Proto()))
print("---------------------------")
print(str(m.param_init_net.Proto()))
print("---------------------------")
workspace.RunNetOnce(m.param_init_net)

workspace.CreateNet(m.net)

for j in range(0, 100):
    data = np.random.rand(16, 100).astype(np.float32)
    label = (np.random.rand(16) * 10).astype(np.int32)
    workspace.FeedBlob("data", data)
    workspace.FeedBlob("label", label)

    workspace.RunNet(m.name, 10)



print(workspace.FetchBlob("softmax"))
print("---------------------------")
print(workspace.FetchBlob("softmax").shape)
print(workspace.FetchBlob("loss"))
