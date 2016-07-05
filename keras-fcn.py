from keras.models import Model
from keras.layers import *
from keras.layers.convolutional import *
import tensorflow as tf
from keras import backend as K

# TODO: are we doing deconvolution the right way?
# TODO: how to incorporate different weight/bias learning rates?
# TODO: other details of FCN that might need to be matched...

# A convolutional block.
# May consist of multiple convolutions of different sizes stacked on top of one another.
# It may also have an optional max-pooling or dropout layer at the end.
def convolution_block(x0, conv_kern_sizes, out_dims, max_pool=1, dropout=0.0, bm='same'):
	# do the stack of convolutional layers
	for i in range(0,len(conv_kern_sizes)):
		k = conv_kern_sizes[i]
		x = Convolution2D(out_dims[i], k, k, border_mode=bm, activation='relu')(x0)
		if dropout > 0.0:  # add a dropout layer in between if required
			x = Dropout(dropout)(x)
		x0 = x
	# add a max-pooling layer at end if required
	if max_pool > 1:
		x = MaxPooling2D(pool_size=(max_pool,max_pool))(x)
	return x

# Crop 'big' to the size of 'small'
def crop_to_tensor(big, small):
	smallc = (small.get_shape()[1])._value
	smallw = (small.get_shape()[2])._value
	smallh = (small.get_shape()[3])._value
	w_diff = (big.get_shape()[2])._value - smallw
	h_diff = (big.get_shape()[3])._value - smallh
	# set sizes and use tensorflow slicing.
	begin = [ 0, 0,w_diff//2,h_diff//2]
	size  = [-1,-1,smallw,smallh]
	# TODO: 
	return Lambda(lambda x: tf.slice(x, begin, size), output_shape=(smallc, smallw, smallh))(big)

# Create a new image cropping layer.
# def crop_tf(x, target_height=0, target_width=0):
# 	return tf.image.resize_image_with_crop_or_pad(x, target_height, target_width)
# def image_crop_outputshape(inpshape, target_h, target_w):
# 	shape = list(inpshape)
# 	assert leng(shape) == 4
# 	shape[2] = target_h
# 	shape[3] = target_w
# 	return tuple(shape)
# def crop_image(inp, height, width):
# 	return Lambda(crop_tf,
# 		          output_shape=lambda sh: image_crop_outputshape(sh,height,width),
# 		          arguments={target_h:height,target_w:width})(inp)

# Takes an input layer, and an input layer from several layers below,
# upsamples the input,
# And combines them by summing.
def upsample2_with_skiplayer(inp, pool):
	# upsample input
	upscore = UpSampling2D(size=(2,2))(inp)
	upscore = Convolution2D(N_labels, 4, 4, border_mode='same', bias=False)(upscore)
	# merge with layer from several layers below.
	score_pool = Convolution2D(N_labels, 1, 1)(pool)
	score_poolc = crop_to_tensor(score_pool, upscore) 
	return merge([score_poolc, upscore], mode='sum')

# Soft-max function on the output.

# TensorFlow version
# The input to this function should be of dimension (nb, channels, width, height)
# The softmax is computed on each (i, :, j, k) vector.
# def array_softmax(inp):
# 	exp_logits = tf.exp(inp)  # compute exponential of each number
# 	sum_exp = tf.reduce_sum(exp_logits, 1, keep_dims=True) # 0, 1, 2, 3
# 	return tf.div(exp_logits, sum_exp)

# Keras version
def array_softmax(inp):
	exp_logits = K.exp(inp)
	sum_exp = K.sum(exp_logits, axis=1, keepdims=True)
	return exp_logits / sum_exp

# Cross-entropy loss on an array of vectors.
# The inputs to this function should be of dimension (nb, channels, width, height).
# First, the cross-entropies are calculated, resulting in (nb, width, height) numbers.
# These are then summed to return the full cross-entropy loss for each input in batch: (nb,)
# y: predictions; y_: one-hot target output classes.
def array_crossentropy(y,y_):
	ce = -tf.reduce_sum(y_ * tf.log(y), 1)
	return tf.reduce_mean(ce, [1, 2])

input_w = 256
N_labels = 60
fin_k_size = 7
in_pad = 16*(fin_k_size-1)
print in_pad

data   = Input(shape=(       1,input_w,input_w))
labels = Input(shape=(N_labels,input_w,input_w))

datap = ZeroPadding2D(padding=(in_pad,in_pad))(data)
# Forward encoding; use convolution blocks.
# There are 5 max-pools so 2^5=32x downsampling.
pool1 = convolution_block(datap, [3, 3],    [ 64,  64], max_pool=2)
pool2 = convolution_block(pool1, [3, 3],    [128, 128], max_pool=2)
pool3 = convolution_block(pool2, [3, 3, 3], [256, 256, 256], max_pool=2)
pool4 = convolution_block(pool3, [3, 3, 3], [512, 512, 512], max_pool=2)
pool5 = convolution_block(pool4, [3, 3, 3], [512, 512, 512], max_pool=2)
fc7   = convolution_block(pool5, [fin_k_size, 1],    [4096, 4096], dropout=0.5, bm='valid') # no padding on this layer
#fc7   = convolution_block(fc6,   [1], [4096], dropout=0.5)
score = convolution_block(fc7,   [1], [N_labels])   # final single 1x1 convolution layer for score layer.

# Now upsample 4x and deconvolution
score_fused = upsample2_with_skiplayer(score,       pool4)
score_final = upsample2_with_skiplayer(score_fused, pool3)

# final 8x upsample and final score
bigscore = UpSampling2D(size=(8,8))(score_final)
bigscore = Convolution2D(N_labels, 16, 16, bias=False, border_mode='same')(bigscore)
score_sf = Lambda(array_softmax, output_shape=(lambda x: x))(bigscore)

print score_sf
# softmax cross-entropy loss
#score_sf = array_softmax(bigscore)

#loss = array_crossentropy(array_softmax(bigscore), labels)
#print loss
model = Model(input=data, output=score_sf)
model.compile(optimizer='sgd', loss='categorical_crossentropy')
