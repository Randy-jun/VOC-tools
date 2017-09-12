#! /usr/bin/env python
# -*- coding: utf-8 -*-
## @package mobilenet_trainer
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import numpy as np
import time
import os
import cv2
import shutil

from caffe2.python import core, workspace, experiment_util, data_parallel_model
from caffe2.python import dyndep, optimizer
from caffe2.python import timeout_guard, model_helper, brew, net_drawer
from caffe2.python import net_drawer

import mobilenet as mobilenet
import caffe2.python.predictor.predictor_exporter as pred_exp
import caffe2.python.predictor.predictor_py_utils as pred_utils
from caffe2.python.predictor_constants import predictor_constants as predictor_constants

'''
Parallelized multi-GPU distributed trainer for mobilenet. Can be used to train
on imagenet data, for example.

To run the trainer in single-machine multi-gpu mode by setting num_shards = 1.

To run the trainer in multi-machine multi-gpu mode with M machines,
run the same program on all machines, specifying num_shards = M, and
shard_id = a unique integer in the set [0, M-1].

For rendezvous (the trainer processes have to know about each other),
you can either use a directory path that is visible to all processes
(e.g. NFS directory), or use a Redis instance. Use the former by
passing the `file_store_path` argument. Use the latter by passing the
`redis_host` and `redis_port` arguments.
'''
logging.basicConfig()
log = logging.getLogger("mobilenet_trainer")
log.setLevel(logging.DEBUG)
dyndep.InitOpsLibrary('@/caffe2/caffe2/distributed:file_store_handler_ops')
dyndep.InitOpsLibrary('@/caffe2/caffe2/distributed:redis_store_handler_ops')

# r_loss = [3]
# r_train_accuracy = [0]
# r_test_accuracy = [0]
# data_base = {'diff_percent_test.lmdb': 12032, 'diff_test.lmdb': 12032}
# data_base = {'testDB_200_quarter_lmdb': 200, 'testDB_200_sub_lmdb': 200, 'testDB_200_top_bottom_lmdb': 200, 'testDB_200_left_right_lmdb': 200}
data_base = {'testDB_quarter_lmdb': 12032, 'testDB_sub_lmdb': 24064, 'testDB_top_bottom_lmdb': 24064, 'testDB_left_right_lmdb': 48128}
# testDB_200_quarter_lmdb
# testDB_200_sub_lmdb
# testDB_200_top_bottom_lmdb
# testDB_200_left_right_lmdb

# data_base = {'diff_percent_test2000.lmdb': 2000, 'diff_test2000.lmdb': 2000}
# for key, v in data_base.items():
#     print(type(key), type(v))
# exit(0)
current_folder = os.path.join(os.path.expanduser("~"), "data/VOCdevkit/dataDB")
data_folder = os.path.join(current_folder)
if not os.path.exists(current_folder):
    print("%s is not exists." % current_folder)
    exit(0)
root_folder = os.path.join(current_folder, 'test_files')

if not os.path.exists(root_folder):
    os.makedirs(root_folder)

# #=========== Set Flag: Flase ============
# flag = False
# np.save("flag.npy", flag)
# #=========== Set Flag: Flase ============

store_folder = os.path.join(current_folder, 'store_files')
if not os.path.exists(store_folder):
    os.makedirs(store_folder)


finetune = os.path.join(store_folder, 'mobilenet_42.mdl')
#=================================================
# if finetune is not None:
#     loa = np.load("result.npz")
#     # np.savez("result.npz", train = test_accuracy, test = test_accuracy)
#     r_train_accuracy = list(loa["train"])
#     r_test_accuracy = list(loa["test"])
#     r_loss = list(loa["loss"])
#=================================================

def AddImageInput(model, reader, batch_size, img_size):
    '''
    Image input operator that loads data from reader and
    applies certain transformations to the images.
    '''
    data, label = brew.image_input(
        model,
        reader, ["data", "label"],
        batch_size=batch_size,
        use_caffe_datum=False,
        mean=128.,
        std=128.,
        minsize=img_size,
        crop=img_size,
    )
    data = model.StopGradient(data, data)
    label = model.StopGradient(label, label)

def CheckSave():
    print("ok")

    #============== Record Data===============
    # np.savez("result.npz", train = r_train_accuracy , test = r_test_accuracy, loss = r_loss)
    #============== Record Data===============
    #============== Break Flag===============
    # flag = np.load("flag.npy")
    # if np.load("flag.npy"):
    #     exit(0)
    #============== Break Flag===============

def LoadModel(path, model):
    '''
    Load pretrained model from file
    '''
    log.info("Loading path: {}".format(path))
    meta_net_def = pred_exp.load_from_db(path, 'minidb')
    init_net = core.Net(pred_utils.GetNet(
        meta_net_def, predictor_constants.GLOBAL_INIT_NET_TYPE))
    predict_init_net = core.Net(pred_utils.GetNet(
        meta_net_def, predictor_constants.PREDICT_INIT_NET_TYPE))

    predict_init_net.RunAllOnGPU()
    init_net.RunAllOnGPU()
    assert workspace.RunNetOnce(predict_init_net)
    assert workspace.RunNetOnce(init_net)


def RunEpoch(
        args,
        db_size,
        test_model,
        batch_size,
):
    '''
    Run one epoch of the trainer.
    TODO: add checkpointing here.
    '''

    epoch_iters = int(db_size / batch_size)
    print(epoch_iters)
    
    test_max_softmax = []
    test_max_index = []
    test_label = []
    # test_max_softmax = np.zeros((epoch_iters), dtype=np.float32)
    # test_max_index = np.zeros((epoch_iters), dtype=np.int)
    # test_label = np.zeros((epoch_iters), dtype=np.int)

    for _ in range(0, epoch_iters):
        workspace.RunNet(test_model.net.Proto().name)
        # print(workspace.FetchBlob('gpu_0/last_out').shape)
        # print(workspace.Blobs())
        # print(workspace.FetchBlob('gpu_0/last_out'))
        # print(workspace.FetchBlob('gpu_0/softmax'))
        # print(workspace.FetchBlob('gpu_0/label'))
        # print(workspace.FetchBlob('gpu_0/loss'))
        # print(workspace.FetchBlob('gpu_0/accuracy'))

        test_softmax = workspace.FetchBlob('gpu_0/softmax')

        # tmp = workspace.Blobs()
        # print(dir(workspace.C.Cursor))
        # print(workspace.C.Cursor.key)
        # print(tmp, workspace.FetchBlob(tmp))
        # exit(0)
        test_max_softmax = np.hstack((test_max_softmax, test_softmax.max(axis=1)))
        test_max_index = np.hstack((test_max_index, test_softmax.argmax(axis=1)))
        test_label = np.hstack((test_label, workspace.FetchBlob('gpu_0/label')))


        # for x in xrange(0, batch_size):
        #     pass
        # print(test_softmax.shape)
        # # print(test_softmax)
        # print(test_softmax.max(axis=1))
        # print(test_softmax.argmax(axis=1))
        # exit(0)

    # flag_accuracy = (test_label == test_max_index).astype(np.int)
    # print(np.mean(flag_accuracy))
    # print(flag_accuracy.shape)
    # print(test_max_softmax.shape)
    # print(test_max_softmax)
    # print(test_max_index.shape)
    # print(test_max_index)
    # print(test_label.shape)
    # print(test_label)
    # print(test_accuracy)
    # print(np.mean(test_accuracy))
    return (test_max_softmax, test_max_index, test_label)


def Test(args):
    # Either use specified device list or generate one
    if args.gpus is not None:
        gpus = [int(x) for x in args.gpus.split(',')]
        num_gpus = len(gpus)
    else:
        gpus = list(range(args.num_gpus))
        num_gpus = args.num_gpus

    log.info("Running on GPUs: {}".format(gpus))
#========================================================
    for db_name, db_size in data_base.items():
        workspace.ResetWorkspace()
        print(workspace.Blobs())

        def create_mobilenet_model_ops(model, loss_scale):
            [softmax, loss] = mobilenet.create_mobilenet(
                model,
                "data",
                num_input_channels=args.num_channels,
                num_labels=args.num_labels,
                label="label"
            )
            # loss = model.Scale(loss, scale=loss_scale)
            brew.accuracy(model, [softmax, "label"], "accuracy")
            # return [loss]

        log.info("----- Create test net ----")
        test_arg_scope = {
            'order': "NCHW",
            'use_cudnn': True,
            'cudnn_exhaustive_search': True,
        }
        test_model = model_helper.ModelHelper(
            name="mobilenet_test", arg_scope=test_arg_scope#, init_params=False
        )

        test_data_db = os.path.join(data_folder, db_name)
        print(test_data_db, db_size)

        test_reader = test_model.CreateDB(
            "test_reader",
            db=test_data_db,
            db_type=args.db_type,
        )

        def test_input_fn(model):
            AddImageInput(
                model,
                test_reader,
                batch_size=args.batch_size,
                img_size=args.image_size,
            )

        data_parallel_model.Parallelize_GPU(
            test_model,
            input_builder_fun=test_input_fn,
            forward_pass_builder_fun=create_mobilenet_model_ops,
            param_update_builder_fun=None,
            devices=gpus,
        )
        workspace.RunNetOnce(test_model.param_init_net)
        workspace.CreateNet(test_model.net)
        # load the pre-trained model and mobilenet epoch

        LoadModel(args.load_model_path, test_model)
        data_parallel_model.FinalizeAfterCheckpoint(test_model) 

        (test_max_softmax, test_max_index, test_label) = RunEpoch(args, db_size, test_model, args.batch_size)
        # flag_accuracy = (test_label == test_max_index).astype(np.int)
        # print(flag_accuracy.mean())

        save_path = os.path.join(root_folder, db_name[:-5] + ".npz")
        print(save_path)
        np.savez(save_path, max_softmax = test_max_softmax , max_index = test_max_index, label = test_label)
        if os.path.exists(save_path):
            print("OK")

def main():
    # TODO: use argv

    # root_folder = os.path.join(current_folder, 'test_files')

    # train_data_db = os.path.join(data_folder, "trainvlaDB_t200_lmdb")
    train_data_db = os.path.join(data_folder, "trainvlaDB_lmdb")
    train_data_db_type = "lmdb"
    train_data_count = 50728
    
    #train_data_count = 50728#50728
    #test_data_count = 12032#12032

    test_data_db = os.path.join(data_folder, "testDB_sub_lmdb")
    test_data_db_type = "lmdb"
    test_data_count = 12032


    parser = argparse.ArgumentParser(
        description="Caffe2: mobilenet training"
    )
    # parser.add_argument("--train_data", type=str, default=train_data_db,
    #                     help="Path to training data or 'everstore_sampler'")
    # parser.add_argument("--test_data", type=str, default=test_data_db,
    #                     help="Path to test data")
    parser.add_argument("--db_type", type=str, default="lmdb",
                        help="Database type (such as lmdb or minidb)")
    parser.add_argument("--gpus", type=str,
                        help="Comma separated list of GPU devices to use")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPU devices (instead of --gpus)")
    parser.add_argument("--num_channels", type=int, default=3,
                        help="Number of color channels")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size (to crop to)")
    parser.add_argument("--num_labels", type=int, default=20,
                        help="Number of labels")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size, total over all GPUs")
    # parser.add_argument("--epoch_size", type=int, default=train_data_count,
    #                     help="Number of images/epoch, total over all machines")
    # parser.add_argument("--num_epochs", type=int, default=200,
    #                     help="Num epochs.")
    # parser.add_argument("--base_learning_rate", type=float, default=0.003,
    #                     help="Initial learning rate.")
    # parser.add_argument("--weight_decay", type=float, default=1e-3,
    #                     help="Weight decay (L2 regularization)")
    parser.add_argument("--cudnn_workspace_limit_mb", type=int, default=64,
                        help="CuDNN workspace limit in MBs")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Number of machines in distributed run")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard id.")
    parser.add_argument("--run_id", type=str,
                        help="Unique run identifier (e.g. uuid)")
    parser.add_argument("--redis_host", type=str,
                        help="Host of Redis server (for rendezvous)")
    parser.add_argument("--redis_port", type=int, default=6379,
                        help="Port of Redis server (for rendezvous)")
    parser.add_argument("--file_store_path", type=str, default=store_folder,
                        help="Path to directory to use for rendezvous")
    parser.add_argument("--save_model_name", type=str, default="mobilenet",
                        help="Save the trained model to a given name")
    parser.add_argument("--load_model_path", type=str, default=finetune,
                        help="Load previously saved model to continue training")
    
    args = parser.parse_args()

    Test(args)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    # from caffe2.python.utils import DebugMode
    # DebugMode.run(main())
    main()
