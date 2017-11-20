import tensorflow as tf

def main():
	basic_cal()

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
	

if __name__ == "__main__":
	main()
