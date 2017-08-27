#! /usr/bin/env python
# -*- coding: utf-8 -*-

# =========== M U S T ===============

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# =========== M U S T ===============
import os
import lmdb
import cv2

from PIL import Image
try:
    import StringIO
except ImportError:
    from io import StringIO

import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, model_helper, brew

Image_width = 224
Image_height = 224
batch_size = 50

def display(env):
	txn = env.begin()
	cur = txn.cursor()
	for k, v in cur:
		print(k, len(v))

def create_data_db(dbpath, img_path, lab):
    db_env = lmdb.open(dbpath, map_size=int(1024*1024*1024*30)) # size:30GB
    # print(db_env.stat())
    # print(dir(db_env.info()))
    # with db_env.begin(write=True) as txn:
    txn = db_env.begin(write=True)
    

    # print(type(tensor_protos))
    # print(dir(tensor_protos))
    for k, value in enumerate(img_path):
        # print("env1")
        # display(db_env)
        # print("env2")
        # print(k)
        lab_v = lab[k]
        # print(k, value, lab_v)
        img = cv2.imread(value, cv2.IMREAD_COLOR).astype(np.uint8)# * (1.0 / 255.0)

        # print(img[0:2, 0:2].shape)

        # cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        img_v = cv2.resize(img, (Image_height, Image_width), interpolation = cv2.INTER_AREA)
        # img_v = img_v.swapaxes(1, 2).swapaxes(0, 1)
        img_v = cv2.cvtColor(img_v,cv2.COLOR_BGR2RGB)

        img_obj = Image.fromarray(img_v)
        img_str = StringIO.StringIO()
        img_obj.save(img_str, 'PNG')
        # print(img_v.shape)
        # flatten_img = img_v.reshape(np.prod(img_v.shape))
        # print(np.array2string(flatten_img))
        # print(type(np.array2string(img_v)))
        tensor_protos = caffe2_pb2.TensorProtos()
        image_tensor = tensor_protos.protos.add()
        # image_tensor.dims.extend(img_v.shape)
        image_tensor.data_type = 4
        image_tensor.string_data.append(img_str.getvalue())
        img_str.close()

        label_tensor = tensor_protos.protos.add()
        label_tensor.data_type = 2
        label_tensor.int32_data.append(lab_v)


        txn.put(
            '{}'.format(k).encode('ascii'),
            tensor_protos.SerializeToString()
        )
        # tensor_protos.Clear()
        if (batch_size - 1) == (k % batch_size):
            txn.commit()
            print("Commit for", k)
            txn = db_env.begin(write=True)

    txn.commit()
    db_env.close()

def read_data_db(dbpath):
    # db_env = lmdb.open(dbpath)#, map_size=int(1024*1024*1024*30)) # size:30GB
    # # print(dir(db_env.info()))
    # # with db_env.begin(write=True) as txn:
    # txn = db_env.begin()
    # display(db_env)
    # exit()
    model = model_helper.ModelHelper(name="lmdbtest")
    # data, label = model.TensorProtosDBInput(
    #     [], ["data", "label"], batch_size = batch_size,
    #     db=dbpath, db_type="lmdb")
    reader = model.CreateDB("test_reader",db=dbpath,db_type="lmdb")
    data, label = brew.image_input(
        model,
        reader, ["data", "label"],
        batch_size=10,
        use_caffe_datum=False,
        mean=128.,
        std=128.,
        minsize=224,
        crop=224,
        mirror=1
    )
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)
    for _ in range(0, 2):
        workspace.RunNet(model.net.Proto().name)
        img_datas = workspace.FetchBlob("data")
        labels = workspace.FetchBlob("label")
        print(labels)
        for k in xrange(0, 10):
            print(img_datas[k].shape)


            # img = img.swapaxes(0, 1).swapaxes(1, 2)
            # cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


def main():
	# print(lmdb.version())
	label_paths = os.path.expanduser("~/data/VOCdevkit/test/label/label_subonly.txt")

	image_paths = []
	labels = []

	with open(label_paths, 'r') as fi:
		lsof = fi.readlines()
		# print(len(lsof))
		for lof in lsof:
			image_paths.append(lof.split()[0])
			labels.append(lof.split()[1].find("1"))
			# labels.append(lof.split()[1])#.count("1"))
		fi.close()

		img_sub = []
		lab_sub = []

		img_top_bottom = []
		lab_top_bottom = []

		img_left_right = []
		lab_left_right = []

		img_quarter = []
		lab_quarter = []


	for k, value in enumerate(image_paths):
		if "sub" == value[-7:-4]:
			img_sub.append(value)
			lab_sub.append(labels[k])
		elif "21" == value[-6:-4] or "22" == value[-6:-4]:
			img_top_bottom.append(value)
			lab_top_bottom.append(labels[k])
		elif "23" == value[-6:-4] or "24" == value[-6:-4]:
			img_left_right.append(value)
			lab_left_right.append(labels[k])
		elif value[-6:-5] == "4":
			img_quarter.append(value)
			lab_quarter.append(labels[k])
		else:
			print("Error!!!")
			exit()

	# print(img_sub)
	# print("==============")
	# print(lab_sub)
	# print("==============")

	# print(img_top_bottom)
	# print("==============")
	# print(lab_top_bottom)
	# print("==============")

	# print(img_left_right)
	# print("==============")
	# print(lab_left_right)
	# print("==============")

	# print(img_quarter)
	# print("==============")
	# print(lab_quarter)
	# print("==============")

	# db_path = os.path.expanduser("~/data/VOCdevkit/dataDB/testDB_200_sub_lmdb")
	db_path = os.path.expanduser("~/data/VOCdevkit/dataDB/testDB_sub_lmdb")
	# create_data_db(db_path, img_sub[:200], lab_sub[:200])
	create_data_db(db_path, img_sub, lab_sub)

	read_data_db(db_path)
	print(db_path + "is OK!")

	# db_path = os.path.expanduser("~/data/VOCdevkit/dataDB/testDB_200_top_bottom_lmdb")
	db_path = os.path.expanduser("~/data/VOCdevkit/dataDB/testDB_top_bottom_lmdb")
	# create_data_db(db_path, img_top_bottom[:200], lab_top_bottom[:200])
	create_data_db(db_path, img_top_bottom, lab_top_bottom)
	read_data_db(db_path)

	print(db_path + " is OK!")

	# db_path = os.path.expanduser("~/data/VOCdevkit/dataDB/testDB_200_left_right_lmdb")
	db_path = os.path.expanduser("~/data/VOCdevkit/dataDB/testDB_left_right_lmdb")
	# create_data_db(db_path, img_left_right[:200], lab_left_right[:200])
	create_data_db(db_path, img_left_right, lab_left_right)
	read_data_db(db_path)

	print(db_path + "is OK!")

	# db_path = os.path.expanduser("~/data/VOCdevkit/dataDB/testDB_200_quarter_lmdb")
	db_path = os.path.expanduser("~/data/VOCdevkit/dataDB/testDB_quarter_lmdb")
	# create_data_db(db_path, img_quarter[:200], lab_quarter[:200])
	create_data_db(db_path, img_quarter, lab_quarter)
	read_data_db(db_path)

	print(db_path + "is OK!")

	# create_data_db(db_path, image_paths, labels)
	# create_data_db(db_path, image_paths[:200], labels[:200])
	# read_data_db(db_path)
	# read_data_db("/home/yroot/data")


if __name__ == '__main__':
	main()