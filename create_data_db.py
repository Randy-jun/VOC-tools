#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
# import shutil
# import random
import lmdb
import cv2
import xml.etree.cElementTree as et

from matplotlib import pyplot

WIDTH = 416
HEIGHT = 416
CHANNEL = 3

def creat_data_lmdb(i_paths, l_paths, dsf_name, width = WIDTH, height = HEIGHT, channel = CHANNEL):
	pyplot.figure()
	size = (width, height)
	for i_path in i_paths:
		print(i_path)
		img = cv2.imread(i_path, cv2.IMREAD_COLOR)
		img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
		
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

datasets = ["trainval", "test"]

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
	print(image_paths[:2])
	print(label_paths[:2])
	dataset_file = os.path.join(root_dir, dataset + "_lmdb")
	os.system('rm -rf ' + dataset_file)

	for k, label_path in enumerate(label_paths[:3]):
		print(k)
		x_label = parseXml(label_path)
		for value in x_label:
			print(value[0], value[1])
	# in_db = lmdb.open(dataset_file)
	# with in_db.begin(write = True) as in_txn:
	# 	for in_idx, image_path in enumerate(image_paths[:30]):
	# 		print(in_idx, image_path)

	# creat_data_lmdb(image_paths[:4], label_paths, dataset_file)