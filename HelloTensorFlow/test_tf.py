from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# traindata = mnist.train
import numpy as np


input_data = tf.random_normal([1,5,5,1])



input = tf.Variable( [4,2,3, 1,5,6] )
y = tf.Variable( [2,2,3, 1,5,6] )

input_a = ( input >3)
input_b = ( y >3)

a = tf.cast(input_a, tf.float32)
b = tf.cast(input_b, tf.float32)
c = tf.cast(tf.equal(input_a, input_b), tf.float32)
c_avg = tf.reduce_mean(tf.cast(tf.equal(input_a, input_b), tf.float32))

v = []
v1 = tf.Variable( [ [1,2],[2,2], [3,2]         ])
v2 = tf.Variable( [ [4,3],[5,3], [6,3]         ])
v.append(v1)
v.append(v2)
v_f = tf.concat(v, 1)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(v1.eval())
    print(v2.eval())
    print(v_f.eval())



# i = tf.train.range_input_producer(10, shuffle=False).dequeue()
# print(i)




