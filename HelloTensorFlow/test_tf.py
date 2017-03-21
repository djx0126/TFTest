from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# traindata = mnist.train
import numpy as np


input_data = tf.random_normal([1,5,5,1])



input = tf.Variable([ [1,2,3],  [4,5,6] ])
i1 = tf.slice(input, [0,0], [-1,2])
i2 = tf.slice(input, [0,2], [-1,1])

input_b = input > 3

b = tf.cast(input_b, tf.float32)

matrix_data_height = 30
filter_sizes = [3,5]
for i,filer_size in enumerate(filter_sizes):
    filer1_height = filer_size
    convd_height = matrix_data_height - filer1_height + 1
    convd_height = int(convd_height/2)
    print(convd_height)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    print(input.eval())
    print(b.eval())
    print(i1.eval())
    print(i2.eval())





