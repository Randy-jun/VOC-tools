#! /usr/bin/env python3
# -*- coding: utf-8 -*-

def create_annoset(root_dir, list_file, out_dir, example_dir, redo = False,\
    anno_type = "detection", label_type = "xml", backend = "lmdb", check_size = False,\
    encode_type = "", encoded = False, gray = False, label_map_file = "", min_dim = 0, max_dim = 0,\
    resize_height = 0, resize_width = 0, shuffle = False, check_label = False):