#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from matplotlib import pyplot

# testDB_sub.npz
# testDB_top_bottom.npz
# testDB_left_right.npz
# testDB_quarter.npz
current_folder = os.path.join(os.path.expanduser("~"), "data/VOCdevkit/dataDB")
root_folder = os.path.join(current_folder, 'test_files')

#=============================================================================================
sub_data_path = os.path.join(root_folder, 'testDB_sub.npz')
sub_load_data = np.load(sub_data_path)
# np.savez(save_path, max_softmax = test_max_softmax , max_index = test_max_index, label = test_label)
sub_max_softmax = sub_load_data["max_softmax"]
sub_max_index = sub_load_data["max_index"]
sub_label = sub_load_data["label"]

sub_flag_accuracy = (sub_label == sub_max_index).astype(np.int)
sub_rev_flag_accuracy = (sub_label != sub_max_index).astype(np.int)

sub_hist_z = sub_flag_accuracy * sub_max_softmax
sub_rev_hist_z = sub_rev_flag_accuracy * sub_max_softmax

sub_hist_index = np.array(np.nonzero(sub_hist_z))
sub_rev_hist_index = np.array(np.nonzero(sub_rev_hist_z))

sub_hist_nz = sub_hist_z[sub_hist_index]
sub_rev_hist_nz = sub_rev_hist_z[sub_rev_hist_index]


# print(sub_flag_accuracy.shape, sub_max_softmax.shape)
# print(sub_hist_z.shape)
# print(sub_hist_index.shape)
# print(sub_hist_nz.shape)

# sub_data = np.zeros((sub_label.shape[0], 3))

# for i in range(0, sub_label.shape[0]):
# 	tmp_perc = per_iD[str(per_diff_index[i])]
# 	tmp_soft = sub_max_softmax[i]
# 	tmp_acc = sub_flag_accuracy[i]
# 	# tmp[0,:] = [tmp_perc, tmp_soft, tmp_acc]
# 	sub_data[i] = [tmp_perc, tmp_soft, tmp_acc]

# sub_count = np.zeros((11, 2))

# for v in sub_data:
#     tmp_i = np.around(10 * v[0]).astype(np.int)
#     # print(v[0])
#     if (v[2]).astype(np.bool):
#         sub_count[tmp_i,0] += 1
#     else:
#         sub_count[tmp_i,1] += 1
# sub_count[9,:] = sub_count[9,:] + sub_count[10,:]
#===============================================================================================
#=============================================================================================
td_data_path = os.path.join(root_folder, 'testDB_top_bottom.npz')
td_load_data = np.load(td_data_path)
# np.savez(save_path, max_softmax = test_max_softmax , max_index = test_max_index, label = test_label)
td_max_softmax = td_load_data["max_softmax"]
td_max_index = td_load_data["max_index"]
td_label = td_load_data["label"]

td_flag_accuracy = (td_label == td_max_index).astype(np.int)
td_rev_flag_accuracy = (td_label != td_max_index).astype(np.int)

td_hist_z = td_flag_accuracy * td_max_softmax
td_rev_hist_z = td_rev_flag_accuracy * td_max_softmax

td_hist_index = np.array(np.nonzero(td_hist_z))
td_rev_hist_index = np.array(np.nonzero(td_rev_hist_z))

td_hist_nz = td_hist_z[td_hist_index]
td_rev_hist_nz = td_rev_hist_z[td_rev_hist_index]


#===============================================================================================
#=============================================================================================
rl_data_path = os.path.join(root_folder, 'testDB_left_right.npz')
rl_load_data = np.load(rl_data_path)
# np.savez(save_path, max_softmax = test_max_softmax , max_index = test_max_index, label = test_label)
rl_max_softmax = rl_load_data["max_softmax"]
rl_max_index = rl_load_data["max_index"]
rl_label = rl_load_data["label"]

rl_flag_accuracy = (rl_label == rl_max_index).astype(np.int)
rl_rev_flag_accuracy = (rl_label != rl_max_index).astype(np.int)

rl_hist_z = rl_flag_accuracy * rl_max_softmax
rl_rev_hist_z = rl_rev_flag_accuracy * rl_max_softmax

rl_hist_index = np.array(np.nonzero(rl_hist_z))
rl_rev_hist_index = np.array(np.nonzero(rl_rev_hist_z))

rl_hist_nz = rl_hist_z[rl_hist_index]
rl_rev_hist_nz = rl_rev_hist_z[rl_rev_hist_index]


#===============================================================================================
#=============================================================================================
sq_data_path = os.path.join(root_folder, 'testDB_quarter.npz')
sq_load_data = np.load(sq_data_path)
# np.savez(save_path, max_softmax = test_max_softmax , max_index = test_max_index, label = test_label)
sq_max_softmax = sq_load_data["max_softmax"]
sq_max_index = sq_load_data["max_index"]
sq_label = sq_load_data["label"]

sq_flag_accuracy = (sq_label == sq_max_index).astype(np.int)
sq_rev_flag_accuracy = (sq_label != sq_max_index).astype(np.int)

sq_hist_z = sq_flag_accuracy * sq_max_softmax
sq_rev_hist_z = sq_rev_flag_accuracy * sq_max_softmax

sq_hist_index = np.array(np.nonzero(sq_hist_z))
sq_rev_hist_index = np.array(np.nonzero(sq_rev_hist_z))

sq_hist_nz = sq_hist_z[sq_hist_index]
sq_rev_hist_nz = sq_rev_hist_z[sq_rev_hist_index]


#===============================================================================================

tdrl_flag_accuracy = np.vstack((td_flag_accuracy[:,np.newaxis], rl_flag_accuracy[:,np.newaxis]))
tdrl_hist_nz = np.hstack((td_hist_nz, rl_hist_nz))
tdrl_rev_hist_nz = np.hstack((td_rev_hist_nz, rl_rev_hist_nz))

#===============================================================================================
fig = pyplot.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

# ax1.bar(np.arange(3)+0.1, np.array([sub_flag_accuracy.mean(), tdrl_flag_accuracy.mean(), sq_flag_accuracy.mean()]).reshape(3,1), 0.2*np.ones((3,1)), np.zeros((3,1)))

# print(sub_hist_nz[0,0:10])
# print(td_hist_nz[0,0:10])
# print(rl_hist_nz[0,0:10])
# print(sq_hist_nz.shape)
ax1.bar(np.arange(3)+0.25, np.array([sub_flag_accuracy.mean(), tdrl_flag_accuracy.mean(), sq_flag_accuracy.mean()]).reshape(3,1), 0.5*np.ones((3,1)), np.zeros((3,1)))
ax1.set_ylim(0, 1)
ax2.hist(sub_rev_hist_nz[0,:], 1000, normed = 1, cumulative=True, alpha=0.85, rwidth=1)
ax2.hist(sub_hist_nz[0,:], 1000, normed = 1, cumulative=True, alpha=0.85, rwidth=1)

# ax2.hist(td_hist_nz[0,:], 100, normed = 1, cumulative=True, alpha=0.85, rwidth=1)
ax3.hist(tdrl_rev_hist_nz[0,:], 1000, normed = 1, cumulative=True, alpha=0.85, rwidth=1)
ax3.hist(tdrl_hist_nz[0,:], 1000, normed = 1, cumulative=True, alpha=0.85, rwidth=1)


ax4.hist(sq_rev_hist_nz[0,:], 1000, normed = 1, cumulative=True, alpha=0.85, rwidth=1)
ax4.hist(sq_hist_nz[0,:], 1000, normed = 1, cumulative=True, alpha=0.85, rwidth=1)


# pyplot.bar(np.arange(4)+0.1, np.array([sub_flag_accuracy.mean(), td_flag_accuracy.mean(), rl_flag_accuracy.mean(), sq_flag_accuracy.mean()]).reshape(4,1), 0.2*np.ones((4,1)), np.zeros((4,1)))
# pyplot.bar(np.arange(9)+0.1, np.array([count[:9,0] / (count[:9,0] + count[:9,1])]).reshape(9,1), 0.2*np.ones((9,1)),np.zeros((9,1)))
# ax1.scatter(range(11), count[:,0] / (count[:,0] + count[:,1]), c='b', marker='o')
# ax2.scatter(data[:, 0], data[:, 1], c='r', marker='o')

# pyplot.ylim([0, 1])
# # pyplot.plot(max_index, "r-.")
pyplot.savefig("test_comp_result.png", dpi = 1200)
os.system("rcp test_comp_result.png yroot@172.18.225.137:/tmp")
pyplot.show()