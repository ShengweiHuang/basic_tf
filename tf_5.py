import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# reference: https://www.tensorflow.org/get_started/mnist/pros

def weight_variable(shape):
	# return random values from a truncated normal distribution
	# stddev: The standard deviation of the truncated normal distribution.
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# define a 2-Dimentional CNN
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# define max pooling layer
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def main():
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	
	# define first convolution neural network
	W_conv1 = weight_variable([5, 5, 1, 32]) # weights, patch size: 5*5, input channels: 1, output channels: 32
	b_conv1 = bias_variable([32]) # biases, let 32 output channels add 32 different biases
	x_image = tf.reshape(x, [-1, 28, 28, 1]) # reshape x, -1 means flatten origin x, let x = 28 * 28 * 1
	# input x_image to CNN, convolution kernal = W_conv1, biases = b_conv1
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	# run max pooling layer, h_pool1 size = 14 * 14 * 32
	h_pool1 = max_pool_2x2(h_conv1)
	
	# define second convolution neural network
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2) # h_pool2 size = 7 * 7 * 64
	
	# define full connected layer
	# let second CNN layer output connected to 10 output
	# each output means the probability of that number
	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # flat h_pool2 to 1-D array
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	# Dropout
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	# Readout layer
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	# Train and Evaluate the Model
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(20000):
			batch = mnist.train.next_batch(50)
			if i % 100 == 0:
				train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
				print('step %d, training accuracy %g' % (i, train_accuracy))
			train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
		print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == "__main__":
	main()
