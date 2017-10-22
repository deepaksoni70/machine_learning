import tensorflow as tf

a=tf.constant(3)
b=tf.constant(4)

with tf.Session() as sess:
    print("Basic Operations : ")
    print("addition with constants : %i" %sess.run(a+b))
    print("multiplication with constants : %i" %sess.run(a*b))

a=tf.placeholder(tf.int16)
b=tf.placeholder(tf.int16)

add=tf.add(a,b)
mul=tf.multiply(a,b)

with tf.Session() as sess:
    print("addition with variables : %i" %sess.run(add, feed_dict={a:12,b:13}))
    print("multiplication with variables : %i" %sess.run(mul, feed_dict={a:12,b:13}))

matrix1=tf.constant([[1,2]])
matrix2=tf.constant([[2],[3]])

product=tf.matmul(matrix1,matrix2)
matrix1=tf.constant([[1,2]])

with tf.Session() as sess:
    result=sess.run(product)
    print("Matrix multiplication result : ")
    print(result)

