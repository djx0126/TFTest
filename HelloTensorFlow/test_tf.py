from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# traindata = mnist.train
import numpy as np


input_data = tf.random_normal([1,5,5,1])
# input_data = tf.constant([[ [ [0.0, 0.0], [1.0, -1.0] ], [[2.0, -2.0], [3.0, -3.0]] ]])
filter_data = tf.random_normal([3,3,1,1])
# filter_data = tf.constant([ [ [ [0.0, 0.0], [1.0, -1.0] ], [ [1.0, -1.0], [-1.0, 1.0] ]  ], [ [ [2,-2], [-2,2] ], [ [3,-3], [-3,3] ] ]   ])



input = tf.Variable(input_data)
filter = tf.Variable(filter_data)

op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
op1 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    print("input")
    print(input.eval())
    print("filter")
    print(filter.eval())
    print("result")
    result = sess.run(op)
    print(result)

    print('///////////////////////////////')
    result = sess.run(op1)
    print(result)

    # filter_data = np.array(filter.eval())
    # input_data = np.array(input.eval()).reshape((2,2,2))
    # filter0 = filter_data.reshape(2,2,2 ,2)[:,:,:,0]
    # patch = input_data[0:2, 0:2]
    # print(filter0)
    # print(patch)
    # print(patch*filter0)
    # print(np.sum(patch*filter0))
    #
    # print('result0 is > ')
    # print(result.reshape((2,2,2))[0][0])


