#! /usr/bin/env python
# -*- coding: utf-8 -*-

# coding: utf-8

# In[12]:

# EXPLAIN WHAT MNIST IS (GIVE EXAMPLE) (SHOW GRAPH STRUCTURE) (MORE VISUALS IN GENERAL)


# # MNIST
# 
# In this tutorial, we will show you how to train an actual CNN model, albeit small. We will be using the old good MNIST dataset and the LeNet model, with a slight change that the sigmoid activations are replaced with ReLUs.
# 
# We will use the model helper - that helps us to deal with parameter initializations naturally.
# 
# First, let's import the necessities.

# In[13]:

# get_ipython().magic(u'matplotlib inline')
from matplotlib import pyplot
import numpy as np
import os
import shutil
import time
# from IPython import display

from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew
from caffe2.proto import caffe2_pb2

# If you would like to see some really detailed initializations,
# you can change --caffe2_log_level=0 to --caffe2_log_level=-1

# workspace.ResetWorkspace()
workspace.GlobalInit(['caffe2', '--caffe2_log_level=1'])

current_folder = os.path.join(os.path.expanduser("~"), "data/VOCdevkit/dataDB")

data_folder = os.path.join(current_folder)
root_folder = os.path.join(current_folder, 'test_files')

train_data_db = os.path.join(data_folder, "trainvlaDB_t200_lmdb")
train_data_db_type = "lmdb"
train_data_count = 200


test_data_db = os.path.join(data_folder, "testDB_200_sub_lmdb")
test_data_db_type = "lmdb"

gpus = [0]
num_labels = 20
batch_size = 10
base_learning_rate = 0.0004 * batch_size

stepsize = int(10 * train_data_count / batch_size)
weight_decay = 1e-4

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

reader = train_model.CreateDB("train_reader", db = train_data_db, db_type = train_data_db_type,)

def AddInput_ops(model):
    # load the data
    data, label = brew.image_input(
    	model,
    	reader,
    	["data", "label"],
    	batch_size = batch_size,
    )
    data = model.StopGradient(data, data)

def TestModel_ops(model, data, label):
    # Image size: 28 x 28 -> 24 x 24
    conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
    # Image size: 24 x 24 -> 12 x 12
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    # Image size: 12 x 12 -> 8 x 8
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
    # Image size: 8 x 8 -> 4 x 4
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
    # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size
    fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
    fc3 = brew.relu(model, fc3, fc3)
    pred = brew.fc(model, fc3, 'pred', 500, 10)
    softmax = brew.softmax(model, pred, 'softmax')
   
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    return [softmax, loss]

def CreateTestModel_ops(model, loss_scale = 1.0):
	[softmax, loss] = TestModel_ops(model, "data", "label")
	prefix = model.net.Proto().name
	loss = model.net.Scale(loss,prefix + "_loos", scale = loss_scale)
	brew.accuracy(model, [softmax, "label"], prefix + "_accuracy")
	return loss


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

exit()

def AddAccuracy(model, softmax, label):
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")
    return accuracy


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

# graph = net_drawer.GetPydotGraphMinimal(
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