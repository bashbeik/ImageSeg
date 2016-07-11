from scipy import misc
import numpy as np
import os 

input_path = '/home/anej001/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
jpgimg_dir = '/home/anej001/data/VOCdevkit/VOC2012/JPEGImages/'
segimg_dir = '/home/anej001/data/VOCdevkit/VOC2012/SegmentationClass/'

width = 256
nlabels = 21

with open(input_path) as f:
	filenames = f.readlines()
	segimg_arr = np.zeros((len(filenames),width,width),  dtype=np.int32)
	img_arr    = np.zeros((len(filenames),width,width,3),dtype=np.float32)
	j=0
	# Load all the files
	for name in filenames:
		# open segmentation image
		path = os.path.join(segimg_dir,"{}{}".format(name.rstrip(), '.png'))
		segimg = misc.imread(path)
		# open color image
		path = os.path.join(jpgimg_dir,"{}{}".format(name.rstrip(), '.jpg'))
		img    = misc.imread(path)
		if segimg.shape[0] > width and segimg.shape[1] > width:
			segimg_arr[j,:,:] = segimg[0:width,0:width]
			img_arr[j,:,:,:] = img[0:width,0:width,:]
			j = j + 1
			#print np.unique(img)
	# Discard unused images
	segimg_arr = segimg_arr[0:j,:,:]
	img_arr    = img_arr[0:j,:,:,:]
	# Convert to one-hot array
	img_onehot = np.zeros((j,width,width,nlabels),dtype=np.float32)
	for i in range(0,j):
		print i
		for x in range(0,width):
			for y in range(0,width):
				labl = segimg_arr[i,x,y]
				if labl==255:
					img_onehot[i,x,y,:] = 1.0/21.0
				else:
					img_onehot[i,x,y,segimg_arr[i,x,y]] = 1.0
	np.save('/home/anej001/data/VOCdevkit/train-label.npy', img_onehot)
	np.save('/home/anej001/data/VOCdevkit/train-data.npy', img_arr)
