#! /usr/bin/env python
# -*- coding: utf-8 -*-

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
from caffe2.python import dyndep, optimizer, utils
from caffe2.python import timeout_guard, model_helper, brew, net_drawer
from caffe2.python.model_helper import ExtractPredictorNet
from caffe2.python import net_drawer
from caffe2.proto import caffe2_pb2

import mobilenet as mobilenet
from caffe2.python.predictor import mobile_exporter as me
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

# a = 'gpu_0/comp_10_spatbn_2_b'
# print(a[6:])
# exit(0)
current_folder = os.path.join(os.path.expanduser("~"), "data/VOCdevkit/dataDB")

if not os.path.exists(current_folder):
    print("%s is not exists." % current_folder)
    exit(0)
root_folder = os.path.join(current_folder, 'test_files')

if not os.path.exists(root_folder):
    os.makedirs(root_folder)

store_folder = os.path.join(current_folder, 'store_files')
if not os.path.exists(store_folder):
    os.makedirs(store_folder)

output_predict_net = os.path.join(store_folder, 'mobilenet_predict_net.pb')
output_init_net = os.path.join(store_folder, 'mobilenet_init_net.pb')

finetune = os.path.join(store_folder, 'mobilenet_8.mdl')

# def AddImageInput(model, reader, batch_size, img_size):
#     '''
#     Image input operator that loads data from reader and
#     applies certain transformations to the images.
#     '''
#     data, label = brew.image_input(
#         model,
#         reader, ["data", "label"],
#         batch_size=batch_size,
#         use_caffe_datum=False,
#         mean=128.,
#         std=128.,
#         minsize=img_size,
#         crop=img_size,
#         mirror=1
#     )
#     data = model.StopGradient(data, data)

def SaveModel(args, test_model):
    prefix = "gpu_0/"
    # print({prefix + str(b): str(b) for b in test_model.params})
    # predictor_export_meta = pred_exp.PredictorExportMeta(
    # print(workspace.Blobs())
    predictor_export_meta = ExtractPredictorNet(
        test_model.net.Proto(),
        # parameters=data_parallel_model.GetCheckpointParams(test_model),
        # parameters=[str(b) for b in test_model.params],
        input_blobs=[prefix + "data"],
        output_blobs=[prefix + "softmax"],
        renames={str(b): str(b)[6:] for b in test_model.params}
    )
    # save the test_model for the current epoch
    model_path = "%s/%s_pred.mdl" % (
        args.file_store_path,
        args.save_model_name,
    )

    # set db_type to be "minidb" instead of "log_file_db", which breaks
    # the serialization in save_to_db. Need to switch back to log_file_db
    # after migration
    print((predictor_export_meta))
    pred_exp.save_to_db(
        db_type="minidb",
        db_destination=model_path,
        predictor_export_meta=predictor_export_meta,
    )

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

def Train(args):
    # Either use specified device list or generate one
    if args.gpus is not None:
        gpus = [int(x) for x in args.gpus.split(',')]
        num_gpus = len(gpus)
    else:
        gpus = list(range(args.num_gpus))
        num_gpus = args.num_gpus

    log.info("Running on GPUs: {}".format(gpus))

    # Verify valid batch size
    total_batch_size = args.batch_size
    batch_per_device = total_batch_size // num_gpus
    assert \
        total_batch_size % num_gpus == 0, \
        "Number of GPUs must divide batch size"

    # Round down epoch size to closest multiple of batch size across machines
    global_batch_size = total_batch_size * args.num_shards
    epoch_iters = int(args.epoch_size / global_batch_size)
    args.epoch_size = epoch_iters * global_batch_size
    log.info("Using epoch size: {}".format(args.epoch_size))

    # Create ModelHelper object
    # train_arg_scope = {
    #     'order': 'NCHW',
    #     'use_cudnn': True,
    #     'cudnn_exhaustice_search': True,
    #     'ws_nbytes_limit': (args.cudnn_workspace_limit_mb * 1024 * 1024),
    # }
    # train_model = model_helper.ModelHelper(
    #     name="mobilenet", arg_scope=train_arg_scope
    # )

    num_shards = args.num_shards

    rendezvous = None

    # Model building functions
    # def create_mobilenet_model_ops(model, loss_scale):
    #     [softmax, loss] = mobilenet.create_mobilenet(
    #         model,
    #         "data",
    #         num_input_channels=args.num_channels,
    #         num_labels=args.num_labels,
    #         label="label",
    #         is_test=True,
    #     )
    #     loss = model.Scale(loss, scale=loss_scale)
    #     brew.accuracy(model, [softmax, "label"], "accuracy")
    #     return [loss]

    # def add_optimizer(model):
    #     stepsz = int(30 * args.epoch_size / total_batch_size / num_shards)
    #     optimizer.add_weight_decay(model, args.weight_decay)
    #     optimizer.build_sgd(
    #         model,
    #         args.base_learning_rate,
    #         momentum=0.9,
    #         nesterov=1,
    #         policy="step",
    #         stepsize=stepsz,
    #         gamma=0.1
    #     )


    # def add_image_input(model):
    #     AddImageInput(
    #         model,
    #         reader,
    #         batch_size=batch_per_device,
    #         img_size=args.image_size,
    #     )
    # def add_post_sync_ops(model):
    #     for param_info in model.GetOptimizationParamInfo(model.GetParams()):
    #         if param_info.blob_copy is not None:
    #             model.param_init_net.HalfToFloat(
    #                 param_info.blob,
    #                 param_info.blob_copy[core.DataType.FLOAT]
    #             )

    test_arg_scope = {
        'order': "NCHW",
        # 'use_cudnn': True,
        # 'cudnn_exhaustive_search': True,
    }
    test_model = model_helper.ModelHelper(
        name="mobilenet_test", arg_scope=test_arg_scope
    )

    deploy_arg_scope = {'order': "NCHW"}
    deploy_model = model_helper.ModelHelper(
        name="mobilenet_deploy", arg_scope=deploy_arg_scope
    )
    mobilenet.create_mobilenet(
        deploy_model,
        "data",
        num_input_channels=args.num_channels,
        num_labels=args.num_labels,
        is_test=True,
    )

    # raw_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    # workspace.FeedBlob("data", raw_data)

    # workspace.RunNetOnce(deploy_model.param_init_net)
    # workspace.CreateNet(deploy_model.net)
    # mobilenet.create_mobilenet(
    #     test_model,
    #     "gpu_0/data",
    #     num_input_channels=args.num_channels,
    #     num_labels=args.num_labels,
    #     is_test=True,
    # )
    # test_reader = test_model.CreateDB(
    #     "test_reader",
    #     db=args.test_data,
    #     db_type=args.db_type,
    # )

    # def test_input_fn(model):
    #     AddImageInput(
    #         model,
    #         test_reader,
    #         batch_size=batch_per_device,
    #         img_size=args.image_size,
    #     )

    # data_parallel_model.Parallelize_GPU(
    #     test_model,
    #     input_builder_fun=test_input_fn,
    #     forward_pass_builder_fun=create_mobilenet_model_ops,
    #     post_sync_builder_fun=add_post_sync_ops,
    #     param_update_builder_fun=None,
    #     devices=gpus,
    # )

    # inputs = np.zeros((32,3,224,224), dtype='f')
    # labels = np.zeros((32,), dtype='f')
    # workspace.FeedBlob("gpu_0/data", inputs)
    # workspace.FeedBlob("gpu_0/label", labels)

    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net)

    
    LoadModel(args.load_model_path, test_model)

    prefix = "gpu_0/"
    for value in deploy_model.params:
        workspace.FeedBlob(value, workspace.FetchBlob(prefix + value))
    # SaveModel(args, test_model)

    # workspace.ResetWorkspace()
    # print(workspace.Blobs())
    # print(deploy_model.params)
    # print("=====================")
    # print(test_model.params)
    # print("=====================")
    # print(workspace.FetchBlob("gpu_0/comp_11_spatbn_2_rm"))
    # print(workspace.FetchBlob("comp_11_spatbn_2_rm"))
    # print(deploy_model.net.Proto())
    # print(deploy_model.param_init_net.Proto())
    # exit(0)

    init_net = caffe2_pb2.NetDef()

    # # print(len(deploy_model.params))
    # # print(deploy_model.param_init_net.Proto())
    # with open("params", 'wb') as f:
    #     f.write(str(deploy_model.param_init_net.Proto()))
    tmp_o = np.zeros((1,1)).astype(np.float32)
    # print(tmp_o.shape)
    # print(type(tmp_o))
    # exit(0)
    init_net.name = "mobilenet_init"
    rm_riv = []
    for value in deploy_model.params:
        tmp = workspace.FetchBlob(prefix + value)
        # print(type(tmp.shape), type(tmp))
        
        if "spatbn" == str(value)[-10:-4]:
            # print(value)
            if "s" == str(value)[-1]:
                # print(str(value)[:-1] + "rm")
                # init_net.op.extend([core.CreateOperator("GivenTensorFill", [], [str(value)[:-1] + "rm"], arg=[utils.MakeArgument("shape", tmp_o.shape), utils.MakeArgument("values", tmp_o)])])
                rm_riv.append(core.CreateOperator("GivenTensorFill", [], [str(value)[:-1] + "rm"], arg=[utils.MakeArgument("shape", tmp_o.shape), utils.MakeArgument("values", tmp_o)]))
                rm_riv.append(core.CreateOperator("GivenTensorFill", [], [str(value)[:-1] + "riv"], arg=[utils.MakeArgument("shape", tmp_o.shape), utils.MakeArgument("values", tmp_o)]))
            # elif "b" == str(value)[-1]:
            #     # print(str(value)[:-1] + "riv")
            #     init_net.op.extend([core.CreateOperator("GivenTensorFill", [], [str(value)[:-1] + "riv"], arg=[utils.MakeArgument("shape", tmp_o.shape), utils.MakeArgument("values", tmp_o)])])
        init_net.op.extend([core.CreateOperator("GivenTensorFill", [], [value], arg=[utils.MakeArgument("shape", tmp.shape), utils.MakeArgument("values", tmp)])])
    init_net.op.extend([core.CreateOperator("ConstantFill", [], ["data"], shape=(1, 3, 224, 224))])
    # exit(0)
    # for value in rm_riv:
        # init_net.op.extend([value])

    deploy_model.net._net.external_output.extend(["softmax"])
    predict_net = deploy_model.net._net

    # print(dir(deploy_model.net._net))
    
    # with open("pparams", 'wb') as f:
    #     f.write(str(deploy_model.param_init_net.Proto()))
    # print(workspace.Blobs())
    # for k, value in enumerate(deploy_model.params):
    #     # print(k,value)
    #     name = k + value
    #     name = workspace.FetchBlob(prefix + value)


    # tmp_work = {value: workspace.FetchBlob(prefix + value) for value in deploy_model.params}
    # # tmp_params = (str(deploy_model.params)


    # workspace.ResetWorkspace()
    # # print(workspace.Blobs())
    # # exit(0)
    # for value in deploy_model.params:
    #     workspace.FeedBlob(value, tmp_work[value])



    # # print(workspace.Blobs())
    # print(workspace.FetchBlob("last_out_b"))
    # exit(0)

    # deploy_model.net._net.external_output.extend(["softmax"])

    # #====================================================================
    # init_net, predict_net = me.Export(workspace, deploy_model.net, deploy_model.params)
    # # print(dir(predict_net.op.remove))
    # # # print(dir(caffe2_pb2.NetDef))
    # # print("===========")
    # # init_net.op.pop(0)
    # flag_di = []
    # print(len(init_net.op))
    # for k, value in enumerate(init_net.op):
    #     for x in value.output:
    #         if ("data" == str(x)) and ("GivenTensorFill" == str(value.type)):
    #             flag_di.append(k)

    # flag_di = sorted(flag_di)
    # for k, v in enumerate(flag_di):
    #     init_net.op.pop(v - k)
    # print(len(init_net.op))

    # flag_dp = []
    # print(len(predict_net.external_input))
    # for k, value in enumerate(predict_net.external_input):
    #     if "data" == str(value):
    #         flag_dp.append(k)

    # flag_dp = sorted(flag_dp)
    # for k, v in enumerate(flag_dp):
    #     predict_net.external_input.pop(v - k)
    
    # print(len(predict_net.external_input))

    # predict_net.external_input.extend(["data"])
    # init_net.op.extend([core.CreateOperator("ConstantFill", [], ["data"], shape=(1, 3, 224, 224))])
    # #==============================================
    
    with open("pred_net", 'wb') as f:
        f.write(str(predict_net))
    # with open("e_pred_net", 'wb') as f:
        # f.write(str(e_predict_net))
    with open("init_net", 'wb') as f:
        f.write(str(init_net))

    with open(output_predict_net, 'wb') as f:
        f.write(predict_net.SerializeToString())
    print(output_predict_net)

    with open(output_init_net, 'wb') as f:
        f.write(init_net.SerializeToString())
    print(output_init_net)

    print("OK!")
    # path = os.path.join(store_folder, 'mobilenet_pred.mdl')
    # print(path)
    # # LoadModel(path, deploy_model)
    # predict_net = pred_exp.prepare_prediction_net(path, "minidb")
    # print(workspace.Blobs())
    # print(workspace.FetchBlob("gpu_0/data"))
    # print(workspace.FetchBlob("gpu_0/softmax"))
    # # inputs = np.zeros((1,3,224,224), dtype='f')
    # # workspace.FeedBlob("gpu_0/data", inputs)
    # workspace.CreateNet(deploy_model.net)
    # workspace.RunNetOnce(deploy_model.net.Proto().name)
    # print(workspace.FetchBlob("gpu_0/data"))
    # print(workspace.FetchBlob("gpu_0/softmax"))
    # print(workspace.Blobs())
    # data_parallel_model.FinalizeAfterCheckpoint(test_model)

    # print(workspace.Blobs())
    # # print(deploy_model.params)
    # (predictor_net, export_blobs) = ExtractPredictorNet(
    #     net_proto=deploy_model.net.Proto(),
    #     input_blobs=["data"],
    #     output_blobs=["softmax"],
    #     device=None,
    #     renames={prefix + str(b): str(b) for b in deploy_model.params}
    # )
    # # print(predictor_net)
    # workspace.ResetWorkspace()
    # print(workspace.Blobs())
    # print(type(predictor_net), type(export_blobs))

    # workspace.RunNet(predictor_net.)
    # print(export_blobs)

    # predictor_export_meta = pred_exp.PredictorExportMeta(
    #     predict_net=test_model.net.Proto(),
    #     parameters=data_parallel_model.GetCheckpointParams(test_model),
    #     inputs=[prefix + "/data"],
    #     outputs=[prefix + "/softmax"],
    #     shapes={
    #         prefix + "/softmax": (1, args.num_labels),
    #         prefix + "/data": (args.num_channels, args.image_size, args.image_size)
    #     }
    # )
    # device_opts = caffe2_pb2.DeviceOption()
    # device_opts.device_type = caffe2_pb2.CPU
    # device_opts.cuda_gpu_id = 0

    # print(test_model.GetParams())

    # img = cv2.imread("./test.jpg", cv2.IMREAD_COLOR)#.astype(np.uint8)
    # img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # img = img[np.newaxis, :, :, :]#.astype(np.float32)
    # img = img.reshape(np.prod(img.shape))

    # # inputs = np.zeros((1,3,224,224), dtype='f')
    # labels = np.ones((1,), dtype='f')
    # workspace.FeedBlob("gpu_0/data", img)
    # workspace.FeedBlob("gpu_0/label", labels)

    # print(workspace.FetchBlob("gpu_0/data"))
    # print(workspace.FetchBlob("gpu_0/label"))
    # workspace.RunNet(test_model.net.Proto().name)
    # print(workspace.FetchBlob("gpu_0/softmax"))

    # print(workspace.Blobs())
    # with open(output_predict_net, 'wb') as f:
    #     f.write(predict_net.SerializeToString())
    # with open(output_init_net, 'wb') as f:
    #     f.write(init_net.SerializeToString())

    # print(output_predict_net)
    # print(output_init_net)



def main():

    data_folder = os.path.join(current_folder)
    test_data_db = os.path.join(data_folder, "testDB_sub_lmdb")

    parser = argparse.ArgumentParser(
        description="Caffe2: mobilenet training"
    )
    parser.add_argument("--train_data", type=str, default=None,
                        help="Path to training data or 'everstore_sampler'")
    parser.add_argument("--test_data", type=str, default=test_data_db,
                        help="Path to test data")
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
    parser.add_argument("--epoch_size", type=int, default=1,
                        help="Number of images/epoch, total over all machines")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="Num epochs.")
    parser.add_argument("--base_learning_rate", type=float, default=0.003,
                        help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help="Weight decay (L2 regularization)")
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

    Train(args)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    # from caffe2.python.utils import DebugMode
    # DebugMode.run(main())
    main()
