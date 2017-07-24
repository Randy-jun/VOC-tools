#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import random
from PIL import Image

def get_image_size(r_dir, ds_file, isn_file):
	with open(ds_file, 'r') as fi:
		lsof = fi.readlines()
		fi.close()
	print("A total if", len(lsof), "images.")
	with open(isn_file, 'w') as fo:
		for lof in lsof:
			img_path = os.path.join(r_dir,lof.split(' ')[0])
			img_size = Image.open(img_path).size
			w_line = os.path.basename(lof.split(' ')[0]).split('.')[0] + " " + img_size[1].__str__() + " " + img_size[0].__str__() + "\n"
			fo.write(w_line)
		fo.close()
	return 0

	# fi.close()
	# print(len(lof)
	# return 0

root_dir = os.environ["HOME"] + "/data/VOCdevkit"
sub_dir = "ImageSets/Main"
script_dir = os.getcwd()

datasets = ["trainval", "test"]
names = ["VOC2007", "VOC2012"]

for dataset in datasets	:
	dst_file = os.path.join(script_dir, dataset + ".txt")
	print(dst_file)
	if os.path.exists(dst_file):
		os.remove(dst_file)
	for name in names:
		if (name == "VOC2012") & (dataset == "test"):
			continue
		
		print("Creating list for", name, dataset)

		dataset_file = os.path.join(root_dir, name, sub_dir, dataset + ".txt")

		img_file = os.path.join(script_dir, dataset + "_img.txt")
		shutil.copy(dataset_file, img_file)
		os.system("sed -i " + "\"s/^/" + name + "\/JPEGImages\//g\" " + img_file)
		os.system("sed -i " + "\"s/$/.jpg/g\" " + img_file)

		label_file = os.path.join(script_dir, dataset + "_label.txt")
		shutil.copy(dataset_file, label_file)
		os.system("sed -i " + "\"s/^/" + name + "\/Annotations\//g\" " + label_file)
		os.system("sed -i " + "\"s/$/.xml/g\" " + label_file)

		os.system("paste -d' ' " + img_file + " " + label_file + " >> " + dst_file)
		os.remove(img_file)
		os.remove(label_file)

	if dataset == "test":
		#print(script_dir + "/get_image_size " + root_dir + " " + dst_file + " " + os.path.join(script_dir, dataset + "_name_size.txt"))
		#os.system(script_dir + "/get_image_size " + root_dir + " " + dst_file + " " + os.path.join(script_dir, dataset + "_name_size.txt"))
		img_size_file = os.path.join(script_dir, dataset + "_name_size.txt")
		get_image_size(root_dir, dst_file, img_size_file)

	if dataset == "trainval":
		with open(dst_file, 'r') as fi:
			lsof = fi.readlines()
			fi.close()
		random.shuffle(lsof)
		with open(dst_file, 'w') as fo:
			fo.writelines(lsof)
			fo.close()
		# with open(isn_file, 'w') as fo: