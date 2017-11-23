import tensorflow as tf
import numpy as np

# define layer
def layer_1(inputs, in_size, out_size, activation_function=None):
	weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	cal_result = tf.matmul(inputs, weights) + biases
	# select activation function, if no activation function, return cal_result
	if activation_function is None:
		outputs = cal_result
	else:
		outputs = activation_function(cal_result)
	return outputs

# define a hidden layer let output = input ^ 2 - input * 2 + 1
# and add multiple layer
def hidden_layer_network():
	xs = tf.placeholder(tf.float32, [None, 1])
	ys = tf.placeholder(tf.float32, [None, 1])
	# add hidden layer
	layer1 = layer_1(xs, 1, 5, activation_function=tf.nn.relu)
	finalLayer = layer_1(layer1, 5, 5, activation_function=tf.nn.relu)
	# add output layer
	prediction = layer_1(finalLayer, 5, 1, activation_function=None)
	# create data and noise
	# create matrix which row=300 col=1 range=-1~1
	x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
	noise = np.random.normal(0, 0.05, x_data.shape)
	# create y let network learning and add noise
	y_data = np.square(x_data) - x_data * 2 + 1 + noise
	# define loss function and select function that reduce loss
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	# initialize
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	# using for loop to train
	for i in range(20001):
		sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
		if i % 1000 == 0:
			# get loss value
			loss_val = sess.run(loss, feed_dict={xs:x_data, ys:y_data})
			print ("i=", i, " loss value=", loss_val)
		if i == 20000:
			# create test data x_test
			x_test = np.linspace(-1, 1, num=10)[:, np.newaxis]
			# get test result and cal real answer
			y_test = sess.run(prediction, feed_dict={xs:x_test})
			y_real = np.square(x_test) - x_test * 2 + 1
			print ("input=", x_test, "real output=", y_real, "output=", y_test, sep="\n")
	sess.close()

if __name__ == "__main__":
	hidden_layer_network()
