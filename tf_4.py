import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# create a simple network to do mnist
# reference: https://www.tensorflow.org/get_started/mnist/beginners
def run_mnist():
	# read mnist data from tensorflow examples
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	# define input x, dimention=1 size=784=(28*28)
	x = tf.placeholder(tf.float32, [None, 784])
	# define 10 different weights and biases
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	# define output y
	y = tf.nn.softmax(tf.matmul(x, W) + b)
	# define correct answer to implement cross-entropy
	y_ = tf.placeholder(tf.float32, [None, 10])
	# define cross entropy
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	# start training
	# define train_step for getting minimize cross_entropy value
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	# initialize
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	# run 1000 iteration
	for i in range(0,1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	# evaluation
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == "__main__":
	run_mnist()
