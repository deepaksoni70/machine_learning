import tensorflow as tf

hello = tf.constant('Hello Tensor FLow')

sess = tf.Session()

print(sess.run(hello))
