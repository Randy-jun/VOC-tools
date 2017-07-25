#! /usr/bin/env python
# -*- coding: utf-8 -*-

from caffe2.python import workspace, model_helper
from caffe2.proto.caffe2_pb2 import NetDef
import numpy as np
exec_data = "./squeezenet/exec_net.pb"
predict_data = "./squeezenet/predict_net.pb"
init_net = NetDef()
init_net.ParseFromString(open(exec_data).read())
predict_net = NetDef()
predict_net.ParseFromString(open(predict_data).read())

# print(predict_net)
predict_net.name = "myfirstnet"
init_net.name = "mynet"

workspace.CreateNet(init_net)
workspace.CreateNet(predict_net)
workspace.RunNet(predict_net.name)
