import tensorflow as tf
import numpy as np

def hello():
	hello = tf.constant("hello, tensorflow")
	sess = tf.Session()
	print(sess.run(hello))

def basic_cal():
	a = tf.constant(5.0)
	b = tf.constant(6.0)
	c = a * b
	sess = tf.Session()
	print(sess.run(c))

def random_float():
	# create 100 random data, value between 0 to 1
	x_data = np.random.rand(100).astype(np.float32)
	# define learning function
	# weight = 0.1
	# biases = 0.3
	y_data = x_data * 0.1 + 0.3
	# create tensorflow structure
	# set weights range and init
	weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
	biases = tf.Variable(tf.zeros([1]))
	# set tensorflow learning function
	y = weights * x_data + biases
	# set loss rule
	loss = tf.reduce_mean(tf.square(y - y_data))
	# set learning oprimizer, learning rate = 0.5
	optimizer = tf.train.GradientDescentOptimizer(0.5)
	# training rule = minimize loss
	train = optimizer.minimize(loss)
	# init variable
	init = tf.initialize_all_variables()
	# end create tensorflow structure
	# start training
	sess = tf.Session()
	# init session
	sess.run(init)
	'''
	using for loop update weight 201 times
	print weight every 20 times
	'''
	for step in range(0, 201):
		sess.run(train)
		if step % 20 == 0:
			print(step, sess.run(weights), sess.run(biases))

if __name__ == "__main__":
	random_float()
