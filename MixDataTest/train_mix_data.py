import tensorflow as tf

from input_data import InputData
from model_fields import ModelFields

sess = tf.InteractiveSession()


model_fields = ModelFields(day_fields=[30, 30, 30, 30, 30],
                           ma_fields=[5, 10, 20, 30, 60],
                           over_all_ma_fields=[5, 10, 20, 30, 60])

matrix_data_width = 5
matrix_data_height = 30
flat_data_length = 10

train_data = InputData(model_fields=model_fields, file_path='train_data_min.txt')
test_data = InputData(model_fields=model_fields, file_path='test_data_min.txt')



input = tf.placeholder(tf.float32, [None, matrix_data_width * matrix_data_height + flat_data_length + 1])
x = tf.slice(input, [0,0], [-1,matrix_data_width * matrix_data_height])
x_f = tf.slice(input, [0,matrix_data_width * matrix_data_height], [-1,flat_data_length])
y_ = tf.slice(input, [0,matrix_data_width * matrix_data_height + flat_data_length], [-1,1])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],
                          strides=[1, 2, 1, 1], padding='VALID')

x_image = tf.reshape(x, [-1,matrix_data_height,matrix_data_width,1]) # M * 30 * 5

# filter_sizes = [3,5]
# pooled_outputs = []
# for i,filer_size in enumerate(filter_sizes):
#     n_filters = 32
#     filter1_height = filer_size
#     W_conv1 = weight_variable([5, matrix_data_width, 1, n_filters])
#     b_conv1 = bias_variable([n_filters])
#     h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)   # M * 26 * 1 * 32
#     convd_height = matrix_data_height - filter1_height + 1
#     h_pool1 = max_pool_2x2(h_conv1)  # M * 13 * 1 * 32
#     pooled_height = int(convd_height/2)
#     h1_flat = tf.reshape(h_pool1, [-1, pooled_height*n_filters])
#     pooled_outputs.append(h1_flat)
#
# pooled_outputs.append(x_f)
# h_flat = tf.concat(pooled_outputs, 1)

n_filters = 64

filter1_height = 5
W_conv1 = weight_variable([5, matrix_data_width, 1, n_filters])
b_conv1 = bias_variable([n_filters])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)   # M * 26 * 1 * 32
convd_height = matrix_data_height - filter1_height + 1
h_pool1 = max_pool_2x2(h_conv1)  # M * 13 * 1 * 32
pooled_height = convd_height/2
h1_flat = tf.reshape(h_pool1, [-1, 13*n_filters])

W_conv2 = weight_variable([3, matrix_data_width, 1, n_filters])
b_conv2 = bias_variable([n_filters])
h_conv2 = tf.nn.relu(conv2d(x_image, W_conv2) + b_conv2) # M * 28 * 1 * 32
h_pool2 = max_pool_2x2(h_conv2)  # M * 14 * 1 * 32
h2_flat = tf.reshape(h_pool2, [-1, 14*n_filters])

h_flat = tf.concat([h1_flat,h2_flat, x_f], 1)

W_fc1 = weight_variable([13*n_filters+14*n_filters+10, 512])
b_fc1 = bias_variable([512])

h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([512, 1])
b_fc2 = bias_variable([1])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

loss = tf.reduce_mean(tf.pow(y_conv-y_, 2))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

target = 3
correct_prediction = tf.equal(y_conv > target, y_ > target)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch = train_data.next(100)

    if i%100 == 0:
        # train_y_output = y_conv.eval(feed_dict={input: batch, keep_prob: 1.0})
        # print(train_y_output.shape)
        # train_y_target = y_.eval(feed_dict={input: batch, keep_prob: 1.0})
        # print(train_y_target.shape)

        train_accuracy = accuracy.eval(feed_dict={input: batch, keep_prob: 1.0})
        train_loss = loss.eval(feed_dict={input: batch, keep_prob: 1.0})
        print("step %d, training accuracy %g, current cost: %g"%(i, train_accuracy, train_loss))

    train_step.run(feed_dict={input: batch, keep_prob: 0.5})

test_accuracy = accuracy.eval(feed_dict={input: test_data.data(), keep_prob: 1.0})
print("test accuracy %g"%(test_accuracy))


