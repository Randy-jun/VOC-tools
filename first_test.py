#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
# import shutil
# import random
import lmdb
import cv2
import xml.etree.cElementTree as et
import numpy as np

from matplotlib import pyplot
from caffe2.python import core, workspace

MODEL = 'squeezenet', 'exec_net.pb', 'predict_net.pb', 227
# MODEL = 'googlenet', 'exec_net.pb', 'predict_net.pb', 224
#mean = 128
INPUT_IMAGE_SIZE = MODEL[3]
#EXEC_NET = os.path.join('/home/yroot/Downloads', MODEL[0], MODEL[1])
#PREDICT_NET = os.path.join('/home/yroot/Downloads', MODEL[0], MODEL[2])

#with open(EXEC_NET) as f:
#	exec_net = f.read()
#with open(PREDICT_NET) as f:
#	predict_net = f.read()

#p = workspace.Predictor(exec_net, predict_net)

def parseXml(file_name):
	result_list = []
	tree = et.parse(file_name)
	root = tree.getroot()
	f_n = root.find("filename")
	# print(f_n.tag + ":" + f_n.text)
	result_list.append((f_n.tag, f_n.text))

	f_ss = root.find("size")
	for f_s in f_ss:
		# print(f_s.tag + ":" + f_s.text)
		result_list.append((f_s.tag, int(f_s.text)))

	f_os = root.findall("object")
	for f_o in f_os:
		f_o_n = f_o.find("name")
		f_o_d = f_o.find("difficult")
		if int(f_o_d.text) == 0:
			result_list.append((f_o_n.tag, f_o_n.text))
			f_o_bs = f_o.find("bndbox")
			temp_xmin = f_o_bs.find("xmin")
			temp_xmax = f_o_bs.find("xmax")
			temp_ymin = f_o_bs.find("ymin")
			temp_ymax = f_o_bs.find("ymax")
			result_list.append((int(temp_xmin.text), int(temp_xmax.text)))
			result_list.append((int(temp_ymin.text), int(temp_ymax.text)))

		# f_o_n = f_o.find("name")
		# # print(f_o_n.tag + ":" + f_o_n.text)
		# result_list.append((f_o_n.tag, f_o_n.text))

		# f_o_d = f_o.find("difficult")
		# # print(f_o_d.tag + ":" + f_o_d.text)
		# result_list.append((f_o_d.tag, int(f_o_d.text)))

		# f_o_bs = f_o.find("bndbox")
		# for f_o_b in f_o_bs:
		# 	# print(f_o_b.tag + ":" + f_o_b.text)
		# 	result_list.append((f_o_b.tag, int(f_o_b.text)))

	return result_list



root_dir = os.environ["HOME"] + "/data/VOCdevkit"
sub_dir = "ImageSets/Main"
script_dir = os.getcwd()

#datasets = ["trainval", "test"]
datasets = ["trainval"]

for dataset in datasets	:
	image_paths = []
	label_paths = []
	dst_file = os.path.join(script_dir, dataset + ".txt")
	print(dst_file)
	with open(dst_file, 'r') as fi:
		lsof = fi.readlines()
		print(len(lsof))
		count = 0
		for lof in lsof:
			image_paths.append(os.path.join(root_dir, lof.split(" ")[0]))
			label_paths.append(os.path.join(root_dir, lof.split(" ")[1])[:-1])
		fi.close()

	for k, label_path in enumerate(label_paths[0:2]):
		image_path = image_paths[k]
		print(image_path, label_path)
		img = cv2.imread(image_path, cv2.IMREAD_COLOR)#.astype(np.float32)

		# pyplot.figure()
		# pyplot.imshow(img)

		cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('img', img)
# 		cv2.waitKey(0)
# 		cv2.destroyAllWindows()

		x_label = parseXml(label_path)

		for k, value in enumerate(x_label):
			print(k, value[0], value[1])
			if (value[0] == 'name'):
				print(x_label[k + 2][0], x_label[k + 2][1], x_label[k + 1][0] ,x_label[k + 1][1])
				# pyplot.figure(k)
				# pyplot.imshow(img[x_label[k + 2][0]:x_label[k + 2][1], x_label[k + 1][0]:x_label[k + 2][1]])
				win_name = x_label[k][1] + str(k)

				# cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
				# cv2.imshow(win_name, img[x_label[k + 2][0]:x_label[k + 2][1], x_label[k + 1][0]:x_label[k + 2][1]])
				
				sub_img = img[x_label[k + 2][0]:x_label[k + 2][1], x_label[k + 1][0]:x_label[k + 1][1]]

				#sub_img = sub_img.astype(np.float32) - mean
				sub_img = cv2.resize(sub_img, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation = cv2.INTER_AREA)

				cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
				cv2.imshow(win_name, sub_img)

				sub_img = sub_img.swapaxes(1, 2).swapaxes(0, 1)
				sub_img = sub_img[np.newaxis, :, :, :].astype(np.float32)

				win_name1 = x_label[k][1] + str(k) + "1"
				cv2.namedWindow(win_name1, cv2.WINDOW_AUTOSIZE)
				cv2.imshow(win_name1, img)
				cv2.waitKey(0)
				cv2.destroyAllWindows()

				# print(img[x_label[k + 2][0]:x_label[k + 2][1], x_label[k + 1][0]:x_label[k + 2][1]].shape)
		# pyplot.show()
#		print(sub_img.shape)
#		sub_img1 = np.arange(10 * 3 * 227 * 227).reshape(10, 3, 227, 227)

	#	print(sub_img1.shape)

	#	for x in xrange(0,10):
	#		print(x)
	#		sub_img1[x] = np.array(sub_img[0])

	#	print(sub_img1.shape)
	#	results = p.run([sub_img1.astype(np.float32)])
	#	results = np.asarray(results)
	#	#results = results[0, 0, :, 0, 0]
	#	print(results.shape)
	#	print(results.argmax(), results.max())
	#	print(results.argmin(), results.min())

		cv2.waitKey(0)
		cv2.destroyAllWindows()

	# img = cv2.imread(image_paths[1], cv2.IMREAD_COLOR)

	# cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
	# cv2.imshow('img', img)
	# cv2.waitKey(0)

	# pyplot.imshow(img[50:250, 50:250])
	# print(img[50:250, 50:250].shape)
	# pyplot.show()

	# in_db = lmdb.open(dataset_file)
	# with in_db.begin(write = True) as in_txn:
	# 	for in_idx, image_path in enumerate(image_paths[:30]):
	# 		print(in_idx, image_path)

	# creat_data_lmdb(image_paths[:4], label_paths, dataset_file)


'''
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
# import shutil
# import random
import lmdb
import cv2
import xml.etree.cElementTree as et
import numpy as np

from matplotlib import pyplot
from caffe2.python import core, workspace

MODEL = 'squeezenet', 'exec_net.pb', 'predict_net.pb', 227
# MODEL = 'googlenet', 'exec_net.pb', 'predict_net.pb', 224
mean = 128
INPUT_IMAGE_SIZE = MODEL[3]
EXEC_NET = os.path.join('/home/yroot/Downloads', MODEL[0], MODEL[1])
PREDICT_NET = os.path.join('/home/yroot/Downloads', MODEL[0], MODEL[2])

with open(EXEC_NET) as f:
	exec_net = f.read()
with open(PREDICT_NET) as f:
	predict_net = f.read()

p = workspace.Predictor(exec_net, predict_net)

def parseXml(file_name):
	result_list = []
	tree = et.parse(file_name)
	root = tree.getroot()
	f_n = root.find("filename")
	# print(f_n.tag + ":" + f_n.text)
	result_list.append((f_n.tag, f_n.text))

	f_ss = root.find("size")
	for f_s in f_ss:
		# print(f_s.tag + ":" + f_s.text)
		result_list.append((f_s.tag, int(f_s.text)))

	f_os = root.findall("object")
	for f_o in f_os:
		f_o_n = f_o.find("name")
		# print(f_o_n.tag + ":" + f_o_n.text)
		result_list.append((f_o_n.tag, f_o_n.text))

		f_o_d = f_o.find("difficult")
		# print(f_o_d.tag + ":" + f_o_d.text)
		result_list.append((f_o_d.tag, int(f_o_d.text)))

		f_o_bs = f_o.find("bndbox")
		for f_o_b in f_o_bs:
			# print(f_o_b.tag + ":" + f_o_b.text)
			result_list.append((f_o_b.tag, int(f_o_b.text)))

	return result_list



root_dir = os.environ["HOME"] + "/data/VOCdevkit"
sub_dir = "ImageSets/Main"
script_dir = os.getcwd()

#datasets = ["trainval", "test"]
datasets = ["test"]

for dataset in datasets	:
	image_paths = []
	label_paths = []
	dst_file = os.path.join(script_dir, dataset + ".txt")
	print(dst_file)
	with open(dst_file, 'r') as fi:
		lsof = fi.readlines()
		print(len(lsof))
		count = 0
		for lof in lsof:
			image_paths.append(os.path.join(root_dir, lof.split(" ")[0]))
			label_paths.append(os.path.join(root_dir, lof.split(" ")[1])[:-1])
		fi.close()

	for k, label_path in enumerate(label_paths[0:1]):
		image_path = image_paths[k]
		print(image_path, label_path)
		img = cv2.imread(image_path, cv2.IMREAD_COLOR)#.astype(np.float32)

		# pyplot.figure()
		# pyplot.imshow(img)

		cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('img', img)
# 		cv2.waitKey(0)
# 		cv2.destroyAllWindows()

		x_label = parseXml(label_path)
		for k, value in enumerate(x_label):
			print(k, value[0], value[1])
			if value[0] == 'name':
				print(x_label[k + 2][1], x_label[k + 4][1], x_label[k + 3][1], x_label[k + 5][1])
				# pyplot.figure(k)
				# pyplot.imshow(img[x_label[k + 2][1]:x_label[k + 4][1], x_label[k + 3][1]:x_label[k + 5][1]])
				win_name = x_label[k][1] + str(k)

				# cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
				# cv2.imshow(win_name, img[x_label[k + 3][1]:x_label[k + 5][1], x_label[k + 2][1]:x_label[k + 4][1]])
				sub_img = img[x_label[k + 3][1]:x_label[k + 5][1], x_label[k + 2][1]:x_label[k + 4][1]]
				#sub_img = sub_img.astype(np.float32) - mean
				sub_img = cv2.resize(sub_img, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation = cv2.INTER_AREA)

				cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
				cv2.imshow(win_name, sub_img)

				sub_img = sub_img.swapaxes(1, 2).swapaxes(0, 1)
				sub_img = sub_img[np.newaxis, :, :, :].astype(np.float32)

				win_name1 = x_label[k][1] + str(k) + "1"
				cv2.namedWindow(win_name1, cv2.WINDOW_AUTOSIZE)
				cv2.imshow(win_name1, img)

				# print(img[x_label[k + 2][1]:x_label[k + 4][1], x_label[k + 3][1]:x_label[k + 5][1]].shape)
		# pyplot.show()
		print(sub_img.shape)
		sub_img1 = np.arange(10 * 3 * 227 * 227).reshape(10, 3, 227, 227)

		print(sub_img1.shape)

		for x in xrange(0,10):
			print(x)
			sub_img1[x] = np.array(sub_img[0])

		print(sub_img1.shape)
		results = p.run([sub_img1.astype(np.float32)])
		results = np.asarray(results)
		#results = results[0, 0, :, 0, 0]
		print(results.shape)
		print(results.argmax(), results.max())
		print(results.argmin(), results.min())

		cv2.waitKey(0)
		cv2.destroyAllWindows()

	# img = cv2.imread(image_paths[1], cv2.IMREAD_COLOR)

	# cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
	# cv2.imshow('img', img)
	# cv2.waitKey(0)

	# pyplot.imshow(img[50:250, 50:250])
	# print(img[50:250, 50:250].shape)
	# pyplot.show()

	# in_db = lmdb.open(dataset_file)
	# with in_db.begin(write = True) as in_txn:
	# 	for in_idx, image_path in enumerate(image_paths[:30]):
	# 		print(in_idx, image_path)

	# creat_data_lmdb(image_paths[:4], label_paths, dataset_file)
'''