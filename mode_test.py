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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
import os
import shutil
import time
# from IPython import display

from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew, memonger
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

train_data_db = os.path.join(data_folder, "trainvlaDB_t200_lmdb")
train_data_db_type = "lmdb"
train_data_count = 200
test_data_count = 200


test_data_db = os.path.join(data_folder, "testDB_200_sub_lmdb")
test_data_db_type = "lmdb"

arg_scope = {"order": "NCHW"}

gpus = [0]
num_labels = 20
batch_size = 5
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
test_model = model_helper.ModelHelper(name = "test")

# reader = train_model.CreateDB("train_reader", db = train_data_db, db_type = train_data_db_type,)
# reader = [train_data_db, train_data_db_type]
def CreateDBReader(reader_db_path, reader_db_type):
	return [reader_db_path, reader_db_type]

def AddInput_ops(model, db_reader):
    # load the dataset
    data, label = model.TensorProtosDBInput(
		[], ["data", "label"], batch_size = batch_size,
		db=db_reader[0], db_type=db_reader[1])
    # data, label = brew.image_input(
    # 	model,
    # 	reader,
    # 	["data", "label"],
    # 	order = "NCHW",
    # 	batch_size = batch_size,
    # 	scale = 227,
    # 	crop = 227,
    # )
    data = model.StopGradient(data, data)

def TestModel_ops(model, data, label):
    # Image size: 227 x 227 -> 224 x 224
    conv1 = brew.conv(model, data, 'conv1', dim_in=3, dim_out=16, kernel=3, weight_init=("MSRAFill", {}))
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    # Image size: 112 x 112 -> 110 x 110
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=16, dim_out=32, kernel=3, weight_init=("MSRAFill", {}))
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)

    # Image size: 55 x 55 -> 52 x 52
    conv3 = brew.conv(model, pool2, 'conv3', dim_in=32, dim_out=64, kernel=3, weight_init=("MSRAFill", {}))
    pool3 = brew.max_pool(model, conv3, 'pool3', kernel=2, stride=2)

    # Image size: 26 x 26 -> 24 x 24
    conv4 = brew.conv(model, pool3, 'conv4', dim_in=64, dim_out=128, kernel=3, weight_init=("MSRAFill", {}))
    pool4 = brew.max_pool(model, conv4, 'pool4', kernel=2, stride=2)

    # Image size: 12 x 12 -> 10 x 10
    conv5 = brew.conv(model, pool4, 'conv5', dim_in=128, dim_out=300, kernel=3, weight_init=("MSRAFill", {}))

    fc1 = brew.fc(model, conv5, 'fc1', dim_in=300 * 10 * 10, dim_out=5000)
    fc1 = brew.relu(model, fc1, fc1)
    pred = brew.fc(model, fc1, 'pred', 5000, 20)

    softmax = brew.softmax(model, pred, 'softmax')
   
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    return [softmax, loss]

# def TestModel_ops(model, data, label):
#     # Image size: 227 x 227 -> 224 x 224
#     conv1 = brew.conv(model, data, 'conv1', dim_in=3, dim_out=64, kernel=3)
#     # conv1 = brew.relu(model, conv1, conv1)
#     conv2 = brew.conv(model, conv1, 'conv2', dim_in=64, dim_out=64, kernel=3)
#     # conv2 = brew.relu(model, conv2, conv2)
#     pool1 = brew.max_pool(model, conv2, 'pool1', kernel=2, stride=2)

#     # Image size: 112 x 112 -> 108 x 108
#     conv3 = brew.conv(model, pool1, 'conv3', dim_in=64, dim_out=128, kernel=3)
#     # conv3 = brew.relu(model, conv3, conv3)
#     conv4 = brew.conv(model, conv3, 'conv4', dim_in=128, dim_out=128, kernel=3)
#     # conv4 = brew.relu(model, conv4, conv4)
#     pool2 = brew.max_pool(model, conv4, 'pool2', kernel=2, stride=2)

# 	# Image size: 54 x 54 -> 50 x 50
#     conv5 = brew.conv(model, pool2, 'conv5', dim_in=128, dim_out=256, kernel=3)
#     # conv5 = brew.relu(model, conv5, conv5)
#     conv6 = brew.conv(model, conv5, 'conv6', dim_in=256, dim_out=256, kernel=3)
#     # conv6 = brew.relu(model, conv6, conv6)
#     pool3 = brew.max_pool(model, conv6, 'pool3', kernel=2, stride=2)

#     # Image size: 25 x 25 -> 20 x 20
#     conv7 = brew.conv(model, pool3, 'conv7', dim_in=256, dim_out=512, kernel=4)
#     # conv7 = brew.relu(model, conv7, conv7)
#     conv8 = brew.conv(model, conv7, 'conv8', dim_in=512, dim_out=512, kernel=3)
#     # conv8 = brew.relu(model, conv8, conv8)
#     pool4 = brew.max_pool(model, conv7, 'pool4', kernel=2, stride=2)

#     # Image size: 10 x 10 -> 6 x 6
#     conv9 = brew.conv(model, pool4, 'conv9', dim_in=512, dim_out=512, kernel=3)
#     # conv9 = brew.relu(model, conv9, conv9)
#     conv10 = brew.conv(model, conv9, 'conv10', dim_in=512, dim_out=512, kernel=3)
#     # conv10 = brew.relu(model, conv10, conv10)

#     fc1 = brew.fc(model, conv10, 'fc1', dim_in=512 * 6 * 6, dim_out=4096)
#     fc1 = brew.relu(model, fc1, fc1)

#     pred = brew.fc(model, fc1, 'pred', 4096, 20)

#     softmax = brew.softmax(model, pred, 'softmax')
   
#     xent = model.LabelCrossEntropy([softmax, label], 'xent')
#     # compute the expected loss
#     loss = model.AveragedLoss(xent, "loss")
#     return [softmax, loss]

def CreateTestModel_ops(model, loss_scale = 1.0):
	[softmax, loss] = TestModel_ops(model, "data", "label")
	# [softmax, loss] = resnet.create_resnet50(model, "data", num_input_channels = 3, num_labels = num_labels, label = "label",)
	prefix = model.net.Proto().name
	# print(dir(model.net.Proto().ListFields()))
	loss = model.net.Scale(loss, prefix + "_loos", scale = loss_scale)
	brew.accuracy(model, [softmax, "label"], prefix + "_accuracy")

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
		namescope = "imonaboat",
		share_activations = False,
	)

def ModelAccuracy(model):
	accuracy = []
	prefix = model.net.Proto().name
	accuracy.append(np.asscalar(workspace.FetchBlob("{}_accuracy".format(prefix))))
	return np.average(accuracy)

device_opt = core.DeviceOption(caffe2_pb2.CUDA, gpus[0])
# with core.NameScope("imonaboat"):
with core.DeviceScope(device_opt):
	reader = CreateDBReader(train_data_db, train_data_db_type)
	AddInput_ops(train_model, reader)
	losses = CreateTestModel_ops(train_model)
	blobs_to_gradients = train_model.AddGradientOperators(losses)
	AddParameterUpdate_ops(train_model)
OptimizeGradientMemory(train_model, [blobs_to_gradients[losses[0]]])

workspace.RunNetOnce(train_model.param_init_net)
workspace.CreateNet(train_model.net, overwrite = True)

with core.DeviceScope(device_opt):
	reader = CreateDBReader(test_data_db, test_data_db_type)
	AddInput_ops(test_model, reader)
	losses = CreateTestModel_ops(test_model)

workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net, overwrite = True)

graph = net_drawer.GetPydotGraphMinimal(
	train_model.net.Proto().op, "test", rankdir = "LR", minimal_dependency = True
)
graph.write_png(os.path.join(root_folder, "train_net.png"))

graph = net_drawer.GetPydotGraphMinimal(
	train_model.param_init_net.Proto().op, "test", rankdir = "LR", minimal_dependency = True
)
graph.write_png(os.path.join(root_folder, "train_init_net.png"))

graph = net_drawer.GetPydotGraphMinimal(
	test_model.net.Proto().op, "test", rankdir = "LR", minimal_dependency = True
)
graph.write_png(os.path.join(root_folder, "test_net.png"))

graph = net_drawer.GetPydotGraphMinimal(
	test_model.param_init_net.Proto().op, "test", rankdir = "LR", minimal_dependency = True
)
graph.write_png(os.path.join(root_folder, "test_int_net.png"))

with open(os.path.join(root_folder, "train_net.pbtxt"), 'w') as fo:
	fo.write(str(train_model.net.Proto()))
with open(os.path.join(root_folder, "train_init_net.pbtxt"), 'w') as fo:
	fo.write(str(train_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "test_net.pbtxt"), 'w') as fo:
	fo.write(str(test_model.net.Proto()))
with open(os.path.join(root_folder, "test_init_net.pbtxt"), 'w') as fo:
	fo.write(str(test_model.param_init_net.Proto()))


############################################

Num_Epochs = 2

############################################
loss = []
train_accuracy = []
test_accuracy = []

for epoch in range(Num_Epochs):
	num_iters = int(train_data_count / batch_size)
	sub_loss = []
	sub_train_accuracy = []
	for iter in range(num_iters):
		t1 = time.time()
		workspace.RunNet(train_model.net.Proto().name)
		t2 = time.time()
		dt = t2 - t1
		sub_loss.append(workspace.FetchBlob("loss"))
		sub_train_accuracy.append(ModelAccuracy(train_model))
		print("train_accurage: %f" % ModelAccuracy(train_model))
		print((
			"Finished iteration {:>" + str(len(str(num_iters))) + "}/{}" +
            " (epoch {:>" + str(len(str(Num_Epochs))) + "}/{})" + 
            " ({:.2f} images/sec)").
            format(iter+1, num_iters, epoch+1, Num_Epochs, batch_size/dt)
        	)
	loss.append(np.average(sub_loss))
	train_accuracy.append(np.average(sub_train_accuracy))

	sub_test_accuracy = []
	for _ in range(int(test_data_count / batch_size)):
		workspace.RunNet(test_model.net.Proto().name)
		sub_test_accuracy.append(ModelAccuracy(test_model))
		print("test_accurage: %f" % ModelAccuracy(test_model))
	
	test_accuracy.append(np.average(sub_test_accuracy))
	print(
		"Train accuracy: {:.3f}, Test accuracy: {:.3f}".
		format(train_accuracy[epoch], test_accuracy[epoch])
		)

print(train_model.Proto())
print(test_model.Proto())

pyplot.figure()
pyplot.plot(loss, 'b')
pyplot.plot(train_accuracy, 'r')
pyplot.plot(test_accuracy, 'g')
pyplot.legend(('Loss', 'Accuracy'), loc='upper right')
pyplot.show()
pyplot.savefig(os.path.join(root_folder, "result.png"), dpi = 600)
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
'''