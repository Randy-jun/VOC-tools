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
# from caffe2.python import core, workspace

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

NameDict = {"person" : 1, "bird": 2, "cat": 3, "cow": 4, "dog": 5, "horse": 6, "sheep": 7, "aeroplane": 8,\
			"bicycle": 9, "boat": 10, "bus": 11, "car": 12, "motorbike": 13, "train": 14, "bottle": 15,\
			"chair": 16, "diningtable": 17, "pottedplant": 18, "sofa": 19, "tvmonitor": 20}

# s_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

s_label = "00000000000000000000"

# tets = "123456789"
# print(tets[:1])
# print(tets[2:])
# exit()
# print(s_label)
# # print(NameDict)	

def setLabel(label, index):
	# print(label)
	# print(len(label), index)
	tmp = int(label[index - 1])

	# tmp += 1 May not need count the number of the object
	tmp = 1
	if (index < 20) & (index > 1):
		label = label[:index - 1] + str(tmp) + label[index:]
	elif 1 == index:
		label = str(tmp) + label[index:]
	elif 20 == index:
		label = label[:index - 1] + str(tmp)
	else:
		print("index fatal")
		exit()
	# print(label)
	# print(len(label))
	return label

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

# datasets = ["trainval", "test"]
datasets = ["trainval"]

for dataset in datasets:
	new_image_path = os.path.join(root_dir, dataset , "image")
	mew_label_path = os.path.join(root_dir, dataset , "label")
	
	if True != os.path.exists(new_image_path):
		os.makedirs(new_image_path)

	if True != os.path.exists(mew_label_path):
		os.makedirs(mew_label_path)


	new_labels = []
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

	for k, label_path in enumerate(label_paths):
		image_path = image_paths[k]
		print(image_path, label_path)
		img = cv2.imread(image_path, cv2.IMREAD_COLOR)#.astype(np.float32)

		# pyplot.figure()
		# pyplot.imshow(img)

		# cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
		# cv2.imshow('img', img)
# 		cv2.waitKey(0)
# 		cv2.destroyAllWindows()

		x_label = parseXml(label_path)

		m_label = s_label[:]
		
		for k, value in enumerate(x_label):
			# print(k, value, value[0], value[1])
			if (value[0] == 'name'):
				# print(x_label[k + 2][0], x_label[k + 2][1], x_label[k + 1][0] ,x_label[k + 1][1])

				tmp_flag = NameDict[value[1]]
				tmp_label = s_label[:]
				m_label = setLabel(m_label, tmp_flag)
				tmp_label = setLabel(tmp_label, tmp_flag)
				
				# pyplot.figure(k)
				# pyplot.imshow(img[x_label[k + 2][0]:x_label[k + 2][1], x_label[k + 1][0]:x_label[k + 2][1]])
				win_name = value[1] + str(k)

				# cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
				# cv2.imshow(win_name, img[x_label[k + 2][0]:x_label[k + 2][1], x_label[k + 1][0]:x_label[k + 2][1]])
				
				sub_img = img[x_label[k + 2][0]:x_label[k + 2][1], x_label[k + 1][0]:x_label[k + 1][1]]

				tmp_file = os.path.join(root_dir, dataset , "image", x_label[0][1][:-4] + win_name + ".jpg")
				# print(os.path.exists(tmp_file))
				if True != os.path.exists(tmp_file):
					cv2.imwrite(tmp_file, sub_img, [int(cv2.IMWRITE_JPEG_QUALITY), 10])

				# tmp_label = os.path.join(root_dir, dataset , "label", x_label[0][1][:-4] + win_name + ".jpg")
				new_labels.append((tmp_file, tmp_label))

				#sub_img = sub_img.astype(np.float32) - mean
				#sub_img = cv2.resize(sub_img, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation = cv2.INTER_AREA)

				# cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
				# cv2.imshow(win_name, sub_img)

				# sub_img = sub_img.swapaxes(1, 2).swapaxes(0, 1)
				# sub_img = sub_img[np.newaxis, :, :, :].astype(np.float32)

				# win_name1 = value[1] + str(k) + "1"
				# cv2.namedWindow(win_name1, cv2.WINDOW_AUTOSIZE)
				# cv2.imshow(win_name1, img)
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()

				# print(img[x_label[k + 2][0]:x_label[k + 2][1], x_label[k + 1][0]:x_label[k + 2][1]].shape)

		# new_labels.append((image_path, m_label)) Just single label classsfier

	# print(new_labels[:])

	with open(os.path.join(root_dir, dataset, "label", "label.txt"), 'w') as fo:
		for k, label in enumerate(new_labels):
			print(str(label))
			fo.write(str(label[0]) + " " + str(label[1])+ "\n")
		fo.close()

	# with open(os.path.join(root_dir, dataset, "label", "label.txt"), 'r') as fi:
	# 	lsof = fi.readlines()
	# 	print(len(lsof[0].split()[1]))
	# 	fi.close()
