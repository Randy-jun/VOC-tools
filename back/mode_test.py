#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# get_ipython().magic(u'matplotlib inline')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
import os
import shutil
import time
import cv2
import mobilenet
# from IPython import display

from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew, memonger, data_parallel_model
from caffe2.python.models import resnet
from caffe2.proto import caffe2_pb2

# If you would like to see some really detailed initializations,
# you can change --caffe2_log_level=0 to --caffe2_log_level=-1

# workspace.ResetWorkspace()
# print(dir(brew.image_input()))
# exit()

workspace.GlobalInit(['caffe2', '--caffe2_log_level=1'])

current_folder = os.path.join(os.path.expanduser("~"), "data/VOCdevkit/dataDB")

data_folder = os.path.join(current_folder)
root_folder = os.path.join(current_folder, 'test_files')

# train_data_db = os.path.join(data_folder, "trainvlaDB_t200_lmdb")
train_data_db = os.path.join(data_folder, "trainvlaDB_lmdb")
train_data_db_type = "lmdb"
train_data_count = 50728
test_data_count = 1200#32

file_store_path = "./mode_save"
save_model_name = "train_mobile_model"

#train_data_count = 50728#50728
#test_data_count = 12032#12032


# test_data_db = os.path.join(data_folder, "testDB_200_sub_lmdb")
test_data_db = os.path.join(data_folder, "testDB_sub_lmdb")
test_data_db_type = "lmdb"

arg_scope = {"order": "NCHW"}

gpus = [0]
num_labels = 20
batch_size = 20
image_size =227
base_learning_rate = 0.003# * batch_size

stepsize = int(5* train_data_count / batch_size)
# stepsize = int(5 * train_data_count / batch_size)
weight_decay = 1e-3

if not os.path.exists(data_folder):
    print("No %s exists." % data_folder)
if os.path.exists(root_folder):
    print("Looks like you ran this before, so we need to cleanup those old files...")
    shutil.rmtree(root_folder)
print(root_folder)
os.makedirs(root_folder)

print("training data folder:" + data_folder)
print("workspace root folder:" + root_folder)

workspace.ResetWorkspace(root_folder)

# In[15]:
train_model = model_helper.ModelHelper(name = "train")
test_model = model_helper.ModelHelper(name = "test")

#=========== Set Flag: Flase ============
flag = False
np.save("flag.npy", flag)
#=========== Set Flag: Flase ============

# reader = train_model.CreateDB("train_reader", db = train_data_db, db_type = train_data_db_type,)
# reader = [train_data_db, train_data_db_type]
def CreateDBReader(reader_db_path, reader_db_type):
    return [reader_db_path, reader_db_type]


# def AddImageInput(model, p_reader, p_batch_size, img_size):
#     '''
#     Image input operator that loads data from reader and
#     applies certain transformations to the images.
#     '''
#     data, label = brew.image_input(
#         model,
#         p_reader, ["data", "label"],
#         batch_size = p_batch_size,
#         use_caffe_datum=True,
#         # mean=128.,
#         # std=128.,
#         scale=img_size,
#         crop=img_size,
#         mirror=1
#     )

#     data = model.StopGradient(data, data)

def AddInput_ops(model):#, db_reader):
    # load the dataset
    data, label = model.TensorProtosDBInput(
        [], ["data", "label"], batch_size = batch_size,
        db=reader[0], db_type=reader[1])

    label = model.StopGradient(label, label)
    data = model.StopGradient(data, data)

# def TestModel_l_ops(model, data, label):
#     # Image size: 227 x 227 -> 224 x 224
#     conv1 = brew.conv(model, data, 'conv1', dim_in=3, dim_out=16, kernel=3, weight_init=("MSRAFill", {}))
#     pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
#     # Image size: 112 x 112 -> 110 x 110
#     conv2 = brew.conv(model, pool1, 'conv2', dim_in=16, dim_out=32, kernel=3, weight_init=("MSRAFill", {}))
#     pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)

#     # Image size: 55 x 55 -> 52 x 52
#     conv3 = brew.conv(model, pool2, 'conv3', dim_in=32, dim_out=64, kernel=3, weight_init=("MSRAFill", {}))
#     pool3 = brew.max_pool(model, conv3, 'pool3', kernel=2, stride=2)

#     # Image size: 26 x 26 -> 24 x 24
#     conv4 = brew.conv(model, pool3, 'conv4', dim_in=64, dim_out=128, kernel=3, weight_init=("MSRAFill", {}))
#     pool4 = brew.max_pool(model, conv4, 'pool4', kernel=2, stride=2)

#     # Image size: 12 x 12 -> 10 x 10
#     conv5 = brew.conv(model, pool4, 'conv5', dim_in=128, dim_out=300, kernel=3, weight_init=("MSRAFill", {}))

#     fc1 = brew.fc(model, conv5, 'fc1', dim_in=300 * 10 * 10, dim_out=5000)
#     fc1 = brew.relu(model, fc1, fc1)
#     pred = brew.fc(model, fc1, 'pred', 5000, 20)

#     softmax = brew.softmax(model, pred, 'softmax')
   
#     xent = model.LabelCrossEntropy([softmax, label], 'xent')
#     # compute the expected loss
#     loss = model.AveragedLoss(xent, "loss")
#     return [softmax, loss]
# ====================== MobileNet ====================
def TestModel_ops(model, data, label):
    # Image size: 227 x 227 -> 224 x 224
    conv1_s2 = brew.conv(model, data, 'conv1_s2', dim_in=3, dim_out=32, kernel=4, stride=2)
    conv1_s2_s = brew.spatial_bn(model, conv1_s2, 'conv1_s2_s', 32, epsilon=1e-3, momentum=0.9)

    conv1_dw = brew.conv(model, conv1_s2_s, 'conv1_dw', dim_in=32, dim_out=32, kernel=3, pad=2)
    # conv1_dw_s = brew.spatial_bn(model, conv1_dw, 'conv1_dw_s', 32, epsilon=1e-3, momentum=0.9)
    conv1_s1 = brew.conv(model, conv1_dw, 'conv1_s1', dim_in=32, dim_out=64, kernel=1)
    conv1_s1_s = brew.spatial_bn(model, conv1_s1, 'conv1_s1_s', 64, epsilon=1e-3, momentum=0.9)

    conv2_s2 = brew.conv(model, conv1_s1_s, 'conv2_s2', dim_in=64, dim_out=64, kernel=3, pad=2, stride=2)
    conv2_dw = brew.conv(model, conv2_s2, 'conv2_dw', dim_in=64, dim_out=128, kernel=1)

    conv2_k3 = brew.conv(model, conv2_dw, 'conv2_k3', dim_in=128, dim_out=128, kernel=3, pad=2)
    conv2_s1 = brew.conv(model, conv2_k3, 'conv2_s1', dim_in=128, dim_out=128, kernel=1)

    conv3_s2 = brew.conv(model, conv2_s1, 'conv3_s2', dim_in=128, dim_out=128, kernel=3, pad=2, stride=2)
    conv3_dw = brew.conv(model, conv3_s2, 'conv3_dw', dim_in=128, dim_out=256, kernel=1)

    conv3_k3 = brew.conv(model, conv3_dw, 'conv3_k3', dim_in=256, dim_out=256, kernel=3, pad=2)
    conv3_s1 = brew.conv(model, conv3_k3, 'conv3_s1', dim_in=256, dim_out=256, kernel=1)

    conv4_s2 = brew.conv(model, conv3_s1, 'conv4_s2', dim_in=256, dim_out=256, kernel=3, pad=2, stride=2)
    conv4_dw = brew.conv(model, conv4_s2, 'conv4_dw', dim_in=256, dim_out=512, kernel=1)
    #5
    conv5_s2 = brew.conv(model, conv4_dw, 'conv5_s2', dim_in=512, dim_out=512, kernel=3, pad=2)
    conv5_dw = brew.conv(model, conv5_s2, 'conv5_dw', dim_in=512, dim_out=512, kernel=1)
    #4
    conv6_s2 = brew.conv(model, conv5_dw, 'conv6_s2', dim_in=512, dim_out=512, kernel=3, pad=2)
    conv6_dw = brew.conv(model, conv6_s2, 'conv6_dw', dim_in=512, dim_out=512, kernel=1)
    #3
    conv7_s2 = brew.conv(model, conv6_dw, 'conv7_s2', dim_in=512, dim_out=512, kernel=3, pad=2)
    conv7_dw = brew.conv(model, conv7_s2, 'conv7_dw', dim_in=512, dim_out=512, kernel=1)
    #2
    conv8_s2 = brew.conv(model, conv7_dw, 'conv8_s2', dim_in=512, dim_out=512, kernel=3, pad=2)
    conv8_dw = brew.conv(model, conv8_s2, 'conv8_dw', dim_in=512, dim_out=512, kernel=1)
    #1
    conv9_s2 = brew.conv(model, conv8_dw, 'conv9_s2', dim_in=512, dim_out=512, kernel=3, pad=2)
    conv9_dw = brew.conv(model, conv9_s2, 'conv9_dw', dim_in=512, dim_out=512, kernel=1)

    conv10_s2 = brew.conv(model, conv9_dw, 'conv10_s2', dim_in=512, dim_out=512, kernel=3, pad=2, stride=2)
    conv10_dw = brew.conv(model, conv10_s2, 'conv10_dw', dim_in=512, dim_out=1024, kernel=1) # out 1024
    conv10_k3 = brew.conv(model, conv10_dw, 'conv10_k3', dim_in=1024, dim_out=1024, kernel=3)
    conv10_s1 = brew.conv(model, conv10_k3, 'conv10_s1', dim_in=1024, dim_out=1024, kernel=1)

    pool1 = brew.average_pool(model, conv10_s1, 'pool1', kernel=7, stride=7)
    # pool1 = brew.max_pool(model, conv10_s1, 'pool1', kernel=7, stride=7)

    fc1 = brew.fc(model, pool1, 'fc1', dim_in=1 * 1 * 1024, dim_out=num_labels)

    # softmax = brew.softmax(model, fc1, 'softmax')

    [softmax, loss] = model.SoftmaxWithLoss([fc1, label], ["softmax", "loss"], )

    return [softmax, loss]
    # xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # # compute the expected loss
    # loss = model.AveragedLoss(xent, "loss")
    # return [softmax, loss]
#====================== MobileNet ====================

#====================== VGG16 ====================
# def TestModel_ops(model, data, label):
#     # Image size: 227 x 227 -> 224 x 224
#     conv1_s2 = brew.conv(model, data, 'conv1_s2', dim_in=3, dim_out=32, kernel=4, stride=2)
#     conv1_dw = brew.conv(model, conv1_s2, 'conv1_dw', dim_in=32, dim_out=32, kernel=3, pad=2)
#     conv1_s1 = brew.conv(model, conv1_dw, 'conv1_s1', dim_in=32, dim_out=64, kernel=1)

#     conv2_s2 = brew.conv(model, conv1_s1, 'conv2_s2', dim_in=64, dim_out=64, kernel=3, pad=2, stride=2)
#     conv2_dw = brew.conv(model, conv2_s2, 'conv2_dw', dim_in=64, dim_out=128, kernel=1)
#     conv2_k3 = brew.conv(model, conv2_dw, 'conv2_k3', dim_in=128, dim_out=128, kernel=3, pad=2)
#     conv2_s1 = brew.conv(model, conv2_k3, 'conv2_s1', dim_in=128, dim_out=128, kernel=1)

#     conv3_s2 = brew.conv(model, conv2_s1, 'conv3_s2', dim_in=128, dim_out=128, kernel=3, pad=2, stride=2)
#     conv3_dw = brew.conv(model, conv3_s2, 'conv3_dw', dim_in=128, dim_out=256, kernel=1)
#     conv3_k3 = brew.conv(model, conv3_dw, 'conv3_k3', dim_in=256, dim_out=256, kernel=3, pad=2)
#     conv3_s1 = brew.conv(model, conv3_k3, 'conv3_s1', dim_in=256, dim_out=256, kernel=1)

#     conv4_s2 = brew.conv(model, conv3_s1, 'conv4_s2', dim_in=256, dim_out=256, kernel=3, pad=2, stride=2)
#     conv4_dw = brew.conv(model, conv4_s2, 'conv4_dw', dim_in=256, dim_out=512, kernel=1)
#     #5
#     conv5_s2 = brew.conv(model, conv4_dw, 'conv5_s2', dim_in=512, dim_out=512, kernel=3, pad=2)
#     conv5_dw = brew.conv(model, conv5_s2, 'conv5_dw', dim_in=512, dim_out=512, kernel=1)
#     #4
#     conv6_s2 = brew.conv(model, conv5_dw, 'conv6_s2', dim_in=512, dim_out=512, kernel=3, pad=2)
#     conv6_dw = brew.conv(model, conv6_s2, 'conv6_dw', dim_in=512, dim_out=512, kernel=1)
#     #3
#     conv7_s2 = brew.conv(model, conv6_dw, 'conv7_s2', dim_in=512, dim_out=512, kernel=3, pad=2)
#     conv7_dw = brew.conv(model, conv7_s2, 'conv7_dw', dim_in=512, dim_out=512, kernel=1)
#     #2
#     conv8_s2 = brew.conv(model, conv7_dw, 'conv8_s2', dim_in=512, dim_out=512, kernel=3, pad=2)
#     conv8_dw = brew.conv(model, conv8_s2, 'conv8_dw', dim_in=512, dim_out=512, kernel=1)
#     #1
#     conv9_s2 = brew.conv(model, conv8_dw, 'conv9_s2', dim_in=512, dim_out=512, kernel=3, pad=2)
#     conv9_dw = brew.conv(model, conv9_s2, 'conv9_dw', dim_in=512, dim_out=512, kernel=1)

#     conv10_s2 = brew.conv(model, conv9_dw, 'conv10_s2', dim_in=512, dim_out=512, kernel=3, pad=2, stride=2)
#     conv10_dw = brew.conv(model, conv10_s2, 'conv10_dw', dim_in=512, dim_out=1024, kernel=1)
#     conv10_k3 = brew.conv(model, conv10_dw, 'conv10_k3', dim_in=1024, dim_out=1024, kernel=3)
#     conv10_s1 = brew.conv(model, conv10_k3, 'conv10_s1', dim_in=1024, dim_out=1024, kernel=1)

#     pool1 = brew.average_pool(model, conv10_s1, 'pool1', kernel=7, stride=7)
#     # pool1 = brew.max_pool(model, conv10_s1, 'pool1', kernel=7, stride=7)

#     fc1 = brew.fc(model, pool1, 'fc1', dim_in=1 * 1 * 1024, dim_out=20)
#     fc1 = brew.relu(model, fc1, fc1)

#     softmax = brew.softmax(model, fc1, 'softmax')
   
#     xent = model.LabelCrossEntropy([softmax, label], 'xent')
#     # compute the expected loss
#     loss = model.AveragedLoss(xent, "loss")
#     return [softmax, loss]
# #====================== VGG16 ====================


def CreateTestModel_ops(model, loss_scale = 1.0):
    [softmax, loss] = TestModel_ops(model, "data", "label")
    # [softmax, loss] = mobilenet.create_mobilenet(model, "data", 3, 20, "label")
    # [softmax, loss] = resnet.create_resnet50(model, "data", num_input_channels = 3, num_labels = num_labels, label = "label",)
    prefix = model.net.Proto().name

    # print(model.net.Proto().ListFields())
    loss = model.Scale(loss, prefix + "_loss", scale = loss_scale)
    model.Accuracy([softmax, "label"], prefix + "_accuracy")
    # brew.accuracy(model, [softmax, "label"], "accuracy")

    return [loss]


def AddParameterUpdate_ops(model):
    brew.add_weight_decay(model, weight_decay)
    iter = brew.iter(model, "iter")
    lr = model.net.LearningRate(
        [iter],
        "lr",
        base_lr = base_learning_rate,
        policy = "step",
        stepsize = stepsize,
        gamma = 0.1,
    )
    for param in model.GetParams():
        param_grad = model.param_to_grad[param]
        param_momentum = model.param_init_net.ConstantFill(
            [param], param + "_momentum", value=0.0
        )

        model.net.MomentumSGDUpdate(
            [param_grad, param_momentum, lr, param],
            [param_grad, param_momentum, param],
            momentum = 0.9,
            nesterov=1,
        )

def OptimizeGradientMemory(model, loss):
    model.net._net = memonger.share_grad_blobs(
        model.net,
        loss,
        set(model.param_to_grad.values()),
        namescope = "test",
        share_activations = False,
    )

def ModelAccuracy(model):
    accuracy = []
    prefix = model.net.Proto().name
    accuracy.append(
        np.asscalar(workspace.FetchBlob("{}_accuracy".format(prefix))))
    return np.average(accuracy)

def CheckSave(s_epoch, s_iter, s_loss, s_train_accuracy):
    print(s_epoch, s_iter)
    sub_test_accuracy = []
    for _ in range(int(test_data_count / batch_size)):
        workspace.RunNet(test_model.net.Proto().name)
        # print("####test#####")
        # print(workspace.FetchBlob("conv2_w"))
        # print("=========")
        # print("####test#####")
        # print(workspace.FetchBlob("conv2_b"))
        # print("=========")
        # img_datas = workspace.FetchBlob("data")
        # for k in xrange(0, batch_size):
        #     img = img_datas[k]
        #     img = img.swapaxes(0, 1).swapaxes(1, 2)
        #     cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
        #     cv2.imshow('img', img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        sub_test_accuracy.append(ModelAccuracy(test_model))
        print("test_accuracy: %f" % ModelAccuracy(test_model))
        # test_accuracy.append(sub_test_accuracy)
    # print(sub_test_accuracy)
    # print("sub_train_accuracy==============")
    # print(len(sub_test_accuracy))

    test_accuracy.append(np.average(sub_test_accuracy))

    #============== Record Data===============
    np.savez("result.npz", train = s_train_accuracy , test = test_accuracy, loss = s_loss)
    # pyplot.figure()
    # pyplot.plot(s_loss, 'b-.')#.')
    # pyplot.plot(s_train_accuracy, 'r--')#o')
    # pyplot.plot(test_accuracy, 'g:')#^')
    # pyplot.legend(('Loss', 'Train_Accuracy', 'Test_Accuracy'), loc='upper right')
    fig = pyplot.figure()
    ax1 = fig.add_subplot(111)
    l_loss, = ax1.plot(loss, 'b-.')#.')
    # ax1.set_ylim([0, max(loss)])
    ax2 = ax1.twinx()
    ax2.set_xlim([0, len(test_accuracy)])
    l_train, = ax2.plot(train_accuracy, 'r--')#o')
    l_test, = ax2.plot(test_accuracy, 'g:')#^')
    ax2.set_ylim([0, 1])

    pyplot.legend((l_loss, l_train, l_test), ('Loss','Train_Accuracy', 'Test_Accuracy'), loc=0)
    pyplot.savefig(os.path.join(root_folder, "result.png"), dpi = 600)
    #============== Record Data===============

    print(
        "Train accuracy: {:.6f}, Test accuracy: {:.6f}".
        format(train_accuracy[-1], test_accuracy[-1])
        )

    #============== Break Flag===============
    flag = np.load("flag.npy")
    if np.load("flag.npy"):
        exit(0)
    #============== Break Flag===============

def SaveModel(train_model, s_iter):
    # prefix = "gpu_{}".format(train_model._devices[0])
    print(train_model.net.Proto())
    print("==============")
    print(data_parallel_model.GetCheckpointParams(train_model))
    predictor_export_meta = pred_exp.PredictorExportMeta(
        predict_net=train_model.net.Proto(),
        parameters=data_parallel_model.GetCheckpointParams(train_model),
        inputs=["data"],
        outputs=["softmax"],
        shapes={
            "softmax": (1, num_labels),
            "data": (3, img_size, img_size)
        }
    )

    # save the train_model for the current epoch
    model_path = "%s/%s_%d.mdl" % (
        file_store_path,
        save_model_name,
        s_iter,
    )
    print(model_path)

    # set db_type to be "minidb" instead of "log_file_db", which breaks
    # the serialization in save_to_db. Need to switch back to log_file_db
    # after migration
    pred_exp.save_to_db(
        db_type="minidb",
        db_destination=model_path,
        predictor_export_meta=predictor_export_meta,
    )
#=======================Strat=========================

reader = CreateDBReader(train_data_db, train_data_db_type)

data_parallel_model.Parallelize_GPU(
        train_model,
        input_builder_fun=AddInput_ops,
        forward_pass_builder_fun=CreateTestModel_ops,
        optimizer_builder_fun=AddParameterUpdate_ops,
        devices=gpus,
        rendezvous=None,
        optimize_gradient_memory=True,
    )

# device_opt = core.DeviceOption(caffe2_pb2.CUDA, gpus[0])
# with core.DeviceScope(device_opt):
#     reader = CreateDBReader(train_data_db, train_data_db_type)
#     AddInput_ops(train_model, reader)
#     # train_reader = train_model.CreateDB(
#     #     "train_reader",
#     #     db=train_data_db,
#     #     db_type=train_data_db_type,
#     # )
#     # AddImageInput(train_model, train_reader, batch_size, 227)
#     losses = CreateTestModel_ops(train_model)
#     blobs_to_gradients = train_model.AddGradientOperators(losses)
#     AddParameterUpdate_ops(train_model)
# OptimizeGradientMemory(train_model, [blobs_to_gradients[losses[0]]])

workspace.RunNetOnce(train_model.param_init_net)
workspace.CreateNet(train_model.net, overwrite = True)

with core.DeviceScope(device_opt):
    reader = CreateDBReader(test_data_db, test_data_db_type)
    AddInput_ops(test_model, reader)
    # test_reader = test_model.CreateDB(
    #     "test_reader",
    #     db=test_data_db,
    #     db_type=test_data_db_type,
    # )
    # AddImageInput(test_model, test_reader, batch_size, 227)
    losses = CreateTestModel_ops(test_model)

workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net, overwrite = True)
'''
graph = net_drawer.GetPydotGraph(
    train_model.net.Proto().op, "test", rankdir = "LR")
graph.write_png(os.path.join(root_folder, "train_net.png"))

graph = net_drawer.GetPydotGraph(
    train_model.param_init_net.Proto().op, "test", rankdir = "LR")
graph.write_png(os.path.join(root_folder, "train_init_net.png"))

graph = net_drawer.GetPydotGraph(
    test_model.net.Proto().op, "test", rankdir = "LR")
graph.write_png(os.path.join(root_folder, "test_net.png"))

graph = net_drawer.GetPydotGraph(
    test_model.param_init_net.Proto().op, "test", rankdir = "LR")
graph.write_png(os.path.join(root_folder, "test_int_net.png"))

with open(os.path.join(root_folder, "train_net.pbtxt"), 'w') as fo:
    fo.write(str(train_model.net.Proto()))
with open(os.path.join(root_folder, "train_init_net.pbtxt"), 'w') as fo:
    fo.write(str(train_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "test_net.pbtxt"), 'w') as fo:
    fo.write(str(test_model.net.Proto()))
with open(os.path.join(root_folder, "test_init_net.pbtxt"), 'w') as fo:
    fo.write(str(test_model.param_init_net.Proto()))
'''


############################################

Num_Epochs = 200

############################################

loss = []
train_accuracy = []
test_accuracy = []

for epoch in range(Num_Epochs):
    sub_loss = []
    sub_train_accuracy = []
    num_iters = int(train_data_count // batch_size)
    s_t = time.time()
    for iter in range(num_iters):

        t1 = time.time()
        workspace.RunNet(train_model.net.Proto().name)
        # print(train_model.GetParams())
        # print(dir(workspace))
        # print("####test#####")
        # print((workspace.FetchBlob("fc1_b")).shape)
        # print("=========")

        t2 = time.time()
        dt = t2 - t1
        # print("####train#####")
        # print(workspace.FetchBlob("fc1").shape)
        # print("======999===")
        # print(workspace.FetchBlob("label"))
        # img_datas = workspace.FetchBlob("data")
        # for k in xrange(0, batch_size):
        #     img = img_datas[k]
        #     img = img.swapaxes(0, 1).swapaxes(1, 2)
        #     cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
        #     cv2.imshow('img', img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # print("=========")

        # print("####train#####")
        # print(workspace.FetchBlob("conv2_b"))
        # print("=========")
        sub_train_accuracy.append(ModelAccuracy(train_model))
        sub_loss.append(np.asscalar(workspace.FetchBlob("train_loss")))
        # loss.append(sub_loss)
        # train_accuracy.append(sub_train_accuracy)
        print("train_accuracy: %f" % ModelAccuracy(train_model))
        print((
            "Finished iteration {:>" + str(len(str(num_iters))) + "}/{}" +
            " (epoch {:>" + str(len(str(Num_Epochs))) + "}/{})" + 
            " ({:.2f} images/sec)").
            format(iter+1, num_iters, epoch+1, Num_Epochs, batch_size/dt)
            )
        if 0 == ((iter + 1) % 10):
            if 0 == ((iter + 1) % 20):
                test_data_count = 12032
                loss.append(np.average(sub_loss))
                train_accuracy.append(np.average(sub_train_accuracy))
                # SaveModel(train_model, iter)
                CheckSave(epoch, iter, loss, train_accuracy)
                test_data_count = 1200
            else:
                loss.append(np.average(sub_loss))
                train_accuracy.append(np.average(sub_train_accuracy))
                CheckSave(epoch, iter, loss, train_accuracy)

    # print(sub_loss)
    # print("sub_loss=============")
    # print(len(sub_loss))
    
    # print(sub_train_accuracy)
    # print("sub_train_accuracy==============")
    # print(len(sub_train_accuracy))
    # loss.append(np.average(sub_loss))
    # train_accuracy.append(np.average(sub_train_accuracy))

    # # sub_test_accuracy = []
    # CheckSave(epoch, loss, train_accuracy)
    # for _ in range(int(test_data_count / batch_size)):
    #     workspace.RunNet(test_model.net.Proto().name)
    #     # print("####test#####")
    #     # print(workspace.FetchBlob("conv2_w"))
    #     # print("=========")
    #     # print("####test#####")
    #     # print(workspace.FetchBlob("conv2_b"))
    #     # print("=========")
    #     # img_datas = workspace.FetchBlob("data")
    #     # for k in xrange(0, batch_size):
    #     #     img = img_datas[k]
    #     #     img = img.swapaxes(0, 1).swapaxes(1, 2)
    #     #     cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
    #     #     cv2.imshow('img', img)
    #     #     cv2.waitKey(0)
    #     #     cv2.destroyAllWindows()
    #     sub_test_accuracy.append(ModelAccuracy(test_model))
    #     print("test_accuracy: %f" % ModelAccuracy(test_model))
    #     # test_accuracy.append(sub_test_accuracy)
    # # print(sub_test_accuracy)
    # # print("sub_train_accuracy==============")
    # # print(len(sub_test_accuracy))

    # test_accuracy.append(np.average(sub_test_accuracy))

    # print(
    #     "Train accuracy: {:.3f}, Test accuracy: {:.3f}".
    #     format(train_accuracy[epoch], test_accuracy[epoch])
    #     )
    e_t = time.time()
    print("This epoch time is {:.3f}s.".format(e_t - s_t))

    
    # print(len(loss), len(train_accuracy), len(test_accuracy))
    #============== Record Data===============
    # np.savez("result.npz", train = test_accuracy, test = train_accuracy, loss = loss)
    # pyplot.figure()
    # pyplot.plot(loss, 'b-.')#.')
    # pyplot.plot(train_accuracy, 'r--')#o')
    # pyplot.plot(test_accuracy, 'g:')#^')
    # pyplot.legend(('Loss', 'Train_Accuracy', 'Test_Accuracy'), loc='upper right')
    # pyplot.savefig(os.path.join(root_folder, "result.png"), dpi = 600)
    # #============== Record Data===============

    # #============== Break Flag===============
    # flag = np.load("flag.npy")
    # if np.load("flag.npy"):
    #     break;
    # #============== Break Flag===============

# print(train_model.Proto())
# print(test_model.Proto())

# np.savez("result.npz", train = test_accuracy, test = test_accuracy, loss = loss)
# pyplot.figure()
# pyplot.plot(loss, 'b')
# pyplot.plot(train_accuracy, 'r')
# pyplot.plot(test_accuracy, 'g')
# pyplot.legend(('Loss', 'Train_Accuracy', 'Test_Accuracy'), loc='upper right')
# # pyplot.show()
# pyplot.savefig(os.path.join(root_folder, "result.png"), dpi = 600)
'''
# def AddAccuracy(model, softmax, label):
#     accuracy = brew.accuracy(model, [softmax, label], "accuracy")
#     return accuracy


# In[18]:

def AddTrainingOperators(model, softmax, label):
    # something very important happens here
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # track the accuracy of the model
    AddAccuracy(model, softmax, label)
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # do a simple stochastic gradient descent
    ITER = brew.iter(model, "iter")
    # set the learning rate schedule
    LR = model.LearningRate(
        ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999 )
    # ONE is a constant value that is used in the gradient update. We only need
    # to create it once, so it is explicitly placed in param_init_net.
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    # Now, for each parameter, we do the gradient updates.
    for param in model.params:
        # Note how we get the gradient of each parameter - CNNModelHelper keeps
        # track of that.
        param_grad = model.param_to_grad[param]
        # The update is a simple weighted sum: param = param + param_grad * LR
        model.WeightedSum([param, ONE, param_grad, LR], param)
    # let's checkpoint every 20 iterations, which should probably be fine.
    # you may need to delete tutorial_files/tutorial-mnist to re-run the tutorial
    model.Checkpoint([ITER] + model.params, [],
                   db="mnist_lenet_checkpoint_%05d.leveldb",
                   db_type="leveldb", every=20)


# In[19]:

arg_scope = {"order": "NCHW"}
train_model = model_helper.ModelHelper(name="mnist_train", arg_scope=arg_scope)
data, label = AddInput(
    train_model, batch_size=64,
    db=os.path.join(data_folder, 'mnist-train-nchw-leveldb'),
    db_type='leveldb')
softmax = AddLeNetModel(train_model, data)
AddTrainingOperators(train_model, softmax, label)

# Testing model. We will set the batch size to 100, so that the testing
# pass is 100 iterations (10,000 images in total).
# For the testing model, we need the data input part, the main LeNetModel
# part, and an accuracy part. Note that init_params is set False because
# we will be using the parameters obtained from the train model.
test_model = model_helper.ModelHelper(
    name="mnist_test", arg_scope=arg_scope, init_params=False)
data, label = AddInput(
    test_model, batch_size=100,
    db=os.path.join(data_folder, 'mnist-test-nchw-leveldb'),
    db_type='leveldb')
softmax = AddLeNetModel(test_model, data)

AddAccuracy(test_model, softmax, label)

# Deployment model. We simply need the main LeNetModel part.
deploy_model = model_helper.ModelHelper(
    name="mnist_deploy", arg_scope=arg_scope, init_params=False)
AddLeNetModel(deploy_model, "data")

# In[20]:

# graph = net_drawer.GetPydotGraph(
    # train_model.net.Proto().op, "mnist", rankdir="LR", minimal_dependency=True)
# display.Image(graph.create_png(), width=800)


# In[21]:

with open(os.path.join(root_folder, "train_net.pbtxt"), 'w') as fid:
    fid.write(str(train_model.net.Proto()))
with open(os.path.join(root_folder, "train_init_net.pbtxt"), 'w') as fid:
    fid.write(str(train_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "test_net.pbtxt"), 'w') as fid:
    fid.write(str(test_model.net.Proto()))
with open(os.path.join(root_folder, "test_init_net.pbtxt"), 'w') as fid:
    fid.write(str(test_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "deploy_net.pbtxt"), 'w') as fid:
    fid.write(str(deploy_model.net.Proto()))
print("Protocol buffers files have been created in your root folder: "+root_folder)


# In[ ]:

# The parameter initialization network only needs to be run once.
workspace.RunNetOnce(train_model.param_init_net)
# creating the network
workspace.CreateNet(train_model.net)
# set the number of iterations and track the accuracy & loss
total_iters = 200    
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)
# Now, we will manually run the network for 200 iterations. 
for i in range(total_iters):
    workspace.RunNet(train_model.net.Proto().name)
    accuracy[i] = workspace.FetchBlob('accuracy')
    loss[i] = workspace.FetchBlob('loss')
    print("Accuracy: %f, Loss: %f" % (accuracy[i], loss[i]))
# After the execution is done, let's plot the values.
pyplot.figure(1)
pyplot.plot(loss, 'b')
pyplot.plot(accuracy, 'r')
pyplot.legend(('Loss', 'Accuracy'), loc='upper right')


# In[14]:

# Let's look at some of the data.
# pyplot.figure()
# data = workspace.FetchBlob('data')
# _ = visualize.NCHW.ShowMultiple(data)
# pyplot.figure()
# softmax = workspace.FetchBlob('softmax')
# _ = pyplot.plot(softmax[0], 'ro')
# pyplot.title('Prediction for the first image')


# In[15]:

# run a test pass on the test net
workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net)
test_accuracy = np.zeros(100)
for i in range(100):
    workspace.RunNet(test_model.net.Proto().name)
    test_accuracy[i] = workspace.FetchBlob('accuracy')
    print("Test_Accuracy: %f" % test_accuracy[i])
# After the execution is done, let's plot the values.
print('Average_Test_Accuracy: %f' % test_accuracy.mean())
pyplot.figure(2)
pyplot.plot(test_accuracy, 'r')
pyplot.title('Acuracy over test batches.')
pyplot.show()
'''
