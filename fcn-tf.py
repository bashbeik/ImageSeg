import tensorflow as tf
import numpy as np

# Convolution layer with bias and activation
# this op performs the following:
# input tensor: [batch, in_height, in_width, in_channels]
# kernel tensor: [filter_height, filter_width, in_channels, out_channels]
def convolution_2d(x0,weights,biases,border_mode):
	x1 = tf.nn.conv2d(x0,weights,border_mode=border_mode)
	x2 = tf.nn.bias_add(x1,biases)
	r  = tf.nn.relu(x2)
	return r

def upsampling_2d(x0,sc):
	w = (x0.get_shape()[2])._value
	h = (x0.get_shape()[3])._value
	return tf.image.resize_images(x0,sc*w,sc*h)

# TODO: are we doing deconvolution the right way?
# TODO: how to incorporate different weight/bias learning rates?
# TODO: other details of FCN that might need to be matched...

# A convolutional block.
# May consist of multiple convolutions of different sizes stacked on top of one another.
# It may also have an optional max-pooling or dropout layer at the end.
def convolution_block(x0, conv_kern_sizes, out_dims, prev_dim, max_pool=1, dropout=0.0, bm='same'):
	# do the stack of convolutional layers
	for i in range(0,len(conv_kern_sizes)):
		k = conv_kern_sizes[i]
		weights = np.random.randn(out_dims[i],prev_dim,k,k)
		biases  = np.random.randn(out_dims[i])
		prev_dim = out_dims[i]
		x = convolution_2d(x0,weights,biases,bm)
		if dropout > 0.0:  # add a dropout layer in between if required
			x = tf.nn.dropout(x,tf.constant(dropout))
		x0 = x
	# add a max-pooling layer at end if required
	if max_pool > 1:
		k = [1, max_pool, max_pool, 1]
		x = tf.nn.max_pool(x, ksize=k, strides=k, padding='SAME')
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
	return tf.slice(big, begin, size)

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
def upsample2_with_skiplayer(inp, pool, weights, biases):
	# upsample input
	upscore = upsampling_2d(inp, 2)
	upscore = tf.nn.conv2d(upscore, weights1, border_mode='SAME')
	# merge with layer from several layers below.
	score_pool = tf.nn.conv2d(pool, weights2, border_mode='VALID')  # weights2: N_labels, 1, 
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
N_labels = 21
fin_k_size = 7
learning_rate = 1e-5

interim_dim = 4096
in_pad = 16*(fin_k_size-1)
print in_pad

#with tf.device('/cpu:0'):
data   = Input(shape=(3,input_w,input_w))
labels = Input(shape=(N_labels,input_w,input_w))

datap = ZeroPadding2D(padding=(in_pad,in_pad))(data)
# Forward encoding; use convolution blocks.
# There are 5 max-pools so 2^5=32x downsampling.
pool1 = convolution_block(datap, [3, 3],    [ 64,  64], 3, max_pool=2)
pool2 = convolution_block(pool1, [3, 3],    [128, 128], 64, max_pool=2)
pool3 = convolution_block(pool2, [3, 3, 3], [256, 256, 256], 128, max_pool=2)
pool4 = convolution_block(pool3, [3, 3, 3], [512, 512, 512], 256, max_pool=2)
pool5 = convolution_block(pool4, [3, 3, 3], [512, 512, 512], 512, max_pool=2)
fc7   = convolution_block(pool5, [fin_k_size, 1],    [interim_dim, interim_dim], 512, dropout=0.5, bm='valid') # no padding on this layer
#fc7   = convolution_block(fc6,   [1], [4096], dropout=0.5)
score = convolution_block(fc7,   [1], [N_labels], interim_dim)   # final single 1x1 convolution layer for score layer.

# Now upsample 4x and deconvolution
score_fused = upsample2_with_skiplayer(score,       pool4)
score_final = upsample2_with_skiplayer(score_fused, pool3)

# final 8x upsample and final score
bigscore = upsampling_2d(score_final, 8)
bigscore = tf.nn.conv2d(bigscore, weights_bigscore, border_mode='SAME') # weights_bigscore: N_labels, 16, 16
score_sf = array_softmax(bigscore)

print score_sf
loss = array_crossentropy(array_softmax(bigscore), labels)

#print score_sf
# softmax cross-entropy loss
#score_sf = array_softmax(bigscore)

print loss

tf.scalar_summary(loss.op.name, loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)
4
#model = Model(input=data, output=score_sf)
#model.compile(optimizer='sgd', loss='categorical_crossentropy')


#-------------------------------------------------------------------------------
import numpy as np
img_data = np.load('train-data.npy', mmap_mode='r')
labl_data = np.load('train-label.npy', mmap_mode='r')

def new_batch(img_data, labl_data, batch_size):
    idx = np.random.randint(0,img_data.shape[0]-1,batch_size)
    return (img_data[idx,:,:,:], labl_data[idx,:,:,:])

# ------------------------------------------------------------------------------

learning_rate = 0.0001
training_epochs = 15
batch_size = 1
display_step = 1
num_examples = img_data.shape[0]

# Define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, use_locking=False, name='GradientDescent').minimize(loss)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
			batch_x, batch_y = new_batch(img_data,labl_data,batch_size)
			batch_x = np.transpose(batch_x, [0,3,2,1])
			batch_y = np.transpose(batch_y, [0,3,2,1])
            # Run optimization op (backprop) and cost op (to get loss value)
			_, c = sess.run([optimizer, loss], feed_dict={data:batch_x, labels:batch_y})
			# Compute average loss
			avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost)
    print "Optimization Finished!"
