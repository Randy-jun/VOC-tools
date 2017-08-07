#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot


loa = np.load("result.npz")

# np.savez("result.npz", train = test_accuracy, test = test_accuracy)
train_accuracy = loa["train"]
test_accuracy = loa["test"]
loss = loa["loss"]
pyplot.figure()
pyplot.plot(loss, 'b')
pyplot.plot(train_accuracy, 'r')
pyplot.plot(test_accuracy, 'g')
pyplot.legend(('loss', 'Train_Accuracy', 'Test_Accuracy'), loc='upper right')
pyplot.show()
#pyplot.savefig(os.path.join(root_folder, "re_result.png"), dpi = 600)