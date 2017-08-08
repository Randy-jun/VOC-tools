#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sys import argv
def test():
	for x in xrange(1,100):
		print(x)
		if np.load("flag.npy"):
			break

	print("test")
		

def main():
	flag = False
	if 1 == int(argv[1]):
		flag = True
	elif 0 == int(argv[1]):
		flag = False
	else:
		print("Argv must be 1 OR 0.")

	np.save("flag.npy", flag)

	# test()

	loa = np.load("flag.npy")

	print("Flag has been set %d" % loa)

if __name__ == '__main__':
	main()