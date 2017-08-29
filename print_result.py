#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from matplotlib import pyplot


loa = np.load("result.npz")

# np.savez("result.npz", train = test_accuracy, test = test_accuracy)
train_accuracy = loa["train"]
test_accuracy = loa["test"]
loss = loa["loss"]

fig = pyplot.figure()

ax1 = fig.add_subplot(111)
# ax1.set_xlim([0, len(test_accuracy)])
l_loss, = ax1.plot(loss, 'b-.')#.')
# ax1.set_ylim([0, max(loss)])
# ax1.legend(('Loss'), loc='upper left')

ax2 = ax1.twinx()
ax2.set_xlim([0, len(test_accuracy)])
l_train, = ax2.plot(train_accuracy, 'r--')#o')
#print(test_accuracy)
l_test, = ax2.plot(test_accuracy, 'g:')#^')
ax2.set_ylim([0, 1])


#pyplot.legend((l_loss, l_train), ('Loss','Train_Accuracy'), loc=0)
pyplot.legend((l_loss, l_train, l_test), ('Loss','Train_Accuracy', 'Test_Accuracy'), loc=0)

pyplot.savefig("result.png", dpi = 1200)
os.system("rcp result.png yroot@172.18.225.137:/tmp")

# pyplot.show()