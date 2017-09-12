#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from matplotlib import pyplot

#diff_test2000.npz
#diff_percent_test2000.npz
current_folder = os.path.join(os.path.expanduser("~"), "data/VOCdevkit/dataDB")
root_folder = os.path.join(current_folder, 'test_files')

#=============================================================================================
per_data_path = os.path.join(root_folder, 'diff_percent_test.npz')
per_load_data = np.load(per_data_path)
# np.savez(save_path, max_softmax = test_max_softmax , max_index = test_max_index, label = test_label)
per_max_softmax = per_load_data["max_softmax"]
per_max_index = per_load_data["max_index"]
per_label = per_load_data["label"]
# print(max_softmax.shape, max_index.shape, label.shape)

per_up_data = np.load("diff_up_index.npy") 
per_diff_index = np.load("diff_percent_count.npy")

per_iD = {str(i[1]): (i[2]).astype(np.float32) for i in per_up_data}

per_flag_accuracy = (per_label == per_max_index)
# flag_accuracy = (label == max_index).astype(np.int)
# print(up_data)
# print(iD)
# iRD = {str(k): int(v) for k, v in diff_index}
# print(diff_index)
# exit(0)
per_data = np.zeros((per_label.shape[0], 3))

for i in range(0, per_label.shape[0]):
	tmp_perc = per_iD[str(per_diff_index[i])]
	tmp_soft = per_max_softmax[i]
	tmp_acc = per_flag_accuracy[i]
	# tmp[0,:] = [tmp_perc, tmp_soft, tmp_acc]
	per_data[i] = [tmp_perc, tmp_soft, tmp_acc]
	# test_max_softmax = np.hstack((test_max_softmax, test_softmax.max(axis=1)))


# flag_accuracy = (label == max_index).astype(np.int)
# print(np.mean(flag_accuracy))
# print(np.mean(max_softmax * flag_accuracy))

# pyplot.plot(max_softmax, "ro")
# pyplot.plot(max_softmax * flag_accuracy, "bo")
# length = len(max_softmax)
# x = np.arange(0, length)
# print(data[:, 0], data[:, 1])

# pyplot.scatter(data[:, 0], data[:, 1], c='b', marker='o')
# pyplot.scatter(data[:, 0], data[:, 1] * data[:, 2], c='b', marker='o')
per_count = np.zeros((11, 2))

for v in per_data:
    tmp_i = np.around(10 * v[0]).astype(np.int)
    # print(v[0])
    if (v[2]).astype(np.bool):
        per_count[tmp_i,0] += 1
    else:
        per_count[tmp_i,1] += 1
per_count[9,:] = per_count[9,:] + per_count[10,:]
#===============================================================================================



#=============================================================================================
data_path = os.path.join(root_folder, 'diff_test.npz')
load_data = np.load(data_path)
# np.savez(save_path, max_softmax = test_max_softmax , max_index = test_max_index, label = test_label)
max_softmax = load_data["max_softmax"]
max_index = load_data["max_index"]
label = load_data["label"]
# print(max_softmax.shape)
down_data = np.load("diff_down_index.npy")
# print(down_data.shape)
diff_index = np.load("diff_count.npy")
# print(diff_index.shape)

iD = {str(i[1]): (i[2]).astype(np.float32) for i in down_data}
flag_accuracy = (label == max_index)

data = np.zeros((label.shape[0], 3))

for i in range(0, label.shape[0]):
    tmp_perc = iD[str(diff_index[i])]
    tmp_soft = max_softmax[i]
    tmp_acc = flag_accuracy[i]
    # tmp[0,:] = [tmp_perc, tmp_soft, tmp_acc]
    data[i] = [tmp_perc, tmp_soft, tmp_acc]
    # test_max_softmax = np.hstack((test_max_softmax, test_softmax.max(axis=1)))


count = np.zeros((11, 2))

for v in data:
    tmp_i = np.around(10 * v[0]).astype(np.int)
    # print(v[0])
    if (v[2]).astype(np.bool):
        count[tmp_i,0] += 1
    else:
        count[tmp_i,1] += 1
count[9,:] = count[9,:] + count[10,:]
#===============================================================================================
# fig = pyplot.figure()
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
pyplot.bar(np.arange(9)+0.1, np.array([per_count[:9,0] / (per_count[:9,0] + per_count[:9,1])]).reshape(9,1), 0.2*np.ones((9,1)),np.zeros((9,1)))
pyplot.bar(np.arange(9)+0.3, np.array([count[:9,0] / (count[:9,0] + count[:9,1])]).reshape(9,1), 0.2*np.ones((9,1)),np.zeros((9,1)))
# pyplot.bar(np.arange(9)+0.2, np.array([count[:9,0] / (count[:9,0] + count[:9,1])]).reshape(9,1), 0.2*np.ones((9,1)),np.zeros((9,1)))
# ax1.scatter(range(11), count[:,0] / (count[:,0] + count[:,1]), c='b', marker='o')
# ax2.scatter(data[:, 0], data[:, 1], c='r', marker='o')

pyplot.ylim([0, 1])
# pyplot.plot(max_index, "r-.")
pyplot.savefig("test_result.png", dpi = 1200)
os.system("rcp test_result.png yroot@172.18.225.137:/tmp")
pyplot.show()
# ax1 = fig.add_subplot(111)
# # ax1.set_xlim([0, len(test_accuracy)])
# l_loss, = ax1.plot(loss, 'b-.')#.')
# # ax1.set_ylim([0, max(loss)])
# # ax1.legend(('Loss'), loc='upper left')

# ax2 = ax1.twinx()
# ax2.set_xlim([0, len(test_accuracy)])
# l_train, = ax2.plot(train_accuracy, 'r--')#o')
# #print(test_accuracy)
# l_test, = ax2.plot(test_accuracy, 'g:')#^')
# ax2.set_ylim([0, 1])


# #pyplot.legend((l_loss, l_train), ('Loss','Train_Accuracy'), loc=0)
# pyplot.legend((l_loss, l_train, l_test), ('Loss','Train_Accuracy', 'Test_Accuracy'), loc=0)

# pyplot.savefig("result.png", dpi = 1200)
# os.system("rcp result.png yroot@172.18.225.137:/tmp")

# # pyplot.show()
