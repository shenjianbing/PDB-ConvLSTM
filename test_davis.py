import numpy as np
import sys
sys.path.append('/your install our caffe path/python')
import caffe
import os
import cv2
import scipy.misc as misc
import datetime
import math

def MaxMinNormalization(x,Max,Min):  
    x = (x - Min) / (Max - Min);  
    return x; 

caffe.set_mode_gpu()
caffe.set_device(0)
print "Load net..."
net = caffe.Net('/your path/test.prototxt', 
	'/your path/pdb-convlstm.caffemodel', caffe.TEST)

with open('/your path/davis_path.txt') as f:
	lines = f.readlines()

for idx in range(len(lines)/5):
	print "Run net..."
	net.forward()
	for i in xrange(5):
		all = net.blobs['conv7_sm'].data
		out = all[i][0]
		line = lines[idx*5+i].replace(" 0\n", "")
		dir, file = os.path.split(line)
		file = file.replace(".jpg", '.png')
		video = dir.split("/")[-1]
		img = misc.imread(line)
		out = misc.imresize(out, img.shape)
		out = out.astype('float')
		out = MaxMinNormalization(out, out.max(), out.min())
		savePath = '/your save path/' + video + '/'
		if not os.path.exists(savePath):
			os.makedirs(savePath)
		misc.imsave(savePath + file, out)
		print 'Save the ' + str(idx*5+i) + ' Image: ' + savePath + file