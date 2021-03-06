import tensorflow as tf

target = 5


class MixDataInput(object):
    def __init__(self, data, batch_size=1, name=None):
        self.name = name
        self.batch_size = batch_size
        self.input_data = data

    def get_fields(self):
        return self.input_data.get_fields()

    def data(self):
        return self.input_data.data()

    def next_batch(self):
        return self.input_data.next(self.batch_size)


class MixDataModel(object):
    def __init__(self, is_training, config, data_input):
        self._is_training = is_training
        self._config = config
        self._data_input = data_input

        matrix_data_width = self._data_input.get_fields().matrix_data_width
        matrix_data_height = self._data_input.get_fields().matrix_data_height
        flat_data_length = self._data_input.get_fields().flat_data_length

        data_width = matrix_data_width * matrix_data_height + flat_data_length + 1
        self._mixed_data = tf.placeholder(tf.float32, [None, data_width], name="input_data")
        x = tf.slice(self._mixed_data, [0, 0], [-1, matrix_data_width * matrix_data_height])
        x_f = tf.slice(self._mixed_data, [0, matrix_data_width * matrix_data_height], [-1, flat_data_length])
        y_ = tf.slice(self._mixed_data, [0, matrix_data_width * matrix_data_height + flat_data_length], [-1, 1])

        x_image = tf.reshape(x, [-1, matrix_data_height, matrix_data_width, 1])  # M * 30 * 5 * 1

        flat_size = 0
        pooled_outputs = []

        conv_input_width = matrix_data_width
        conv_input_height = matrix_data_height
        conv_input_filters = 1
        cnn_x = x_image
        for i, cnn_layer_filters in enumerate(self._config.cnn_filters):
            with tf.name_scope("conv_layer" + str(i)):
                num_layer_filters = cnn_layer_filters[0]
                conv_size = cnn_layer_filters[1]
                num_filters = cnn_layer_filters[2]

                for j in range(num_layer_filters):
                    scope_name = "conv_layer" + str(i) + "_conv_" + str(j)
                    with tf.name_scope(scope_name):
                        W_conv1 = self.weight_variable([conv_size, conv_input_width, conv_input_filters, num_filters],
                                                       scope_name)
                        tf.add_to_collection('losses', tf.nn.l2_loss(W_conv1))
                        b_conv1 = self.bias_variable([num_filters], scope_name)
                        h_conv1 = tf.nn.relu(self.conv2d(cnn_x, W_conv1) + b_conv1)  # M * 26 * 1 * 32
                        conv1_height = conv_input_height
                        if self._config.padding_valid:
                            conv1_height = conv_input_height - conv_size + 1  # padding=VALID

                        cnn_x = h_conv1
                        conv_input_height = conv1_height
                        conv_input_width = 1
                        conv_input_filters = num_filters

                pool_output = self.max_pool_2x2(cnn_x)  # M * 13 * 1 * 32
                pooled_output_height = conv_input_height // 2
                cnn_x = pool_output
                conv_input_height = pooled_output_height

        partial_flat_size = conv_input_height * conv_input_filters
        flat_size += partial_flat_size
        h1_flat = tf.reshape(cnn_x, [-1, partial_flat_size])
        self._keep_prob = tf.placeholder(tf.float32)
        h1_flat_drop = tf.nn.dropout(h1_flat, self._keep_prob)
        pooled_outputs.append(h1_flat_drop)

        if flat_data_length > 0:
            pooled_outputs.append(x_f)
            flat_size += flat_data_length
        h_flat = tf.concat(pooled_outputs, 1)

        nc_1_size = self._config.fc_size
        with tf.name_scope("fc1"):
            W_fc1 = self.weight_variable([flat_size, nc_1_size], "fc1")
            tf.add_to_collection('losses', tf.nn.l2_loss(W_fc1))
            b_fc1 = self.bias_variable([nc_1_size], "fc1")
            h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, self._keep_prob)

        with tf.name_scope("fc2"):
            W_fc2 = self.weight_variable([nc_1_size, 1], "fc2")
            tf.add_to_collection('losses', tf.nn.l2_loss(W_fc2))
            b_fc2 = self.bias_variable([1], "fc2")
            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        self._cost = tf.reduce_mean(tf.pow(y_conv - y_, 2) + config.beta * total_loss)
        self._train_op = tf.train.AdamOptimizer(1e-4).minimize(self._cost)

        correct_prediction = tf.equal(y_conv > target, y_ > target)
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self._b = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

        target_buy = tf.reduce_sum(tf.cast(y_conv > target, tf.float32))
        self._idea_target_buy = tf.reduce_sum(tf.cast(y_ > target, tf.float32))
        self._target_buy_correct = tf.reduce_sum(tf.cast((y_conv > target) & (y_ > target), tf.float32))
        self._buy_success = self._target_buy_correct / target_buy

    def weight_variable(self, shape, scope_name):
        with tf.variable_scope(scope_name):
            w = tf.get_variable("w", shape, dtype=tf.float32)
            # print("variable w name = " + w.name + ", reuse?" + str(tf.get_variable_scope().reuse))
            return w

    def bias_variable(self, shape, scope_name):
        with tf.variable_scope(scope_name):
            return tf.get_variable("b", shape, dtype=tf.float32)

    def conv2d(self, x, W):
        padding = 'VALID'
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

    def max_pool_2x2(self, x):
        padding = 'VALID'
        return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],
                              strides=[1, 2, 1, 1], padding=padding)

    @property
    def is_training(self):
        return self._is_training

    @property
    def data(self):
        return self._data_input

    @property
    def mixed_data(self):
        return self._mixed_data

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def train_op(self):
        return self._train_op

    @property
    def cost(self):
        return self._cost

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def buy_success(self):
        return self._buy_success

    @property
    def idea_target_buy(self):
        return self._idea_target_buy

    @property
    def b(self):
        return self._b
