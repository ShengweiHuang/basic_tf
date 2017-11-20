import tensorflow as tf

def hello():
	hello = tf.constant("hello, tensorflow")
	sess = tf.Session()
	print(sess.run(hello))
	sess.close()

def basic_cal():
	a = tf.constant(5.0)
	b = tf.constant(6.0)
	c = a * b
	sess = tf.Session()
	print(sess.run(c))
	sess.close()

if __name__ == "__main__":
	hello()
	basic_cal()
