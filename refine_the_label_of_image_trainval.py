#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

label_path = os.path.expanduser("~/data/VOCdevkit/trainval/label/label.txt")

test = 1
test1 = 1

with open(label_path, 'r') as fi:
	fi_labels = fi.readlines()
	for fi_label in fi_labels:
		if fi_label.split()[1].count("1") != 1:
			print(test)
			test += 1
		else:
			print(test1)
			test1 += 1
	fi.close()

# with open(os.path.join(root_dir, dataset, "label", "label.txt"), 'r') as fi:
# 	lsof = fi.readlines()
# 	print(len(lsof[0].split()[1]))
# 	fi.close()
