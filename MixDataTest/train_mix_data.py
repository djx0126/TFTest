import tensorflow as tf

from input_data import InputData
from model_fields import ModelFields
import configs

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")

FLAGS = flags.FLAGS

model_fields = ModelFields(day_fields=[120, 0, 0, 0, 0],
                           ma_fields=[],
                           over_all_ma_fields=[])

train_file = 'train120'
test_file = 'test120'

matrix_data_width = 1
matrix_data_height = 120
flat_data_length = 0

target = 5

def get_config():
    if FLAGS.model == "small":
        print("use small config")
        return configs.SmallConfig()
    elif FLAGS.model == "medium":
        print("use medium config")
        return configs.MidConfig()
    elif FLAGS.model == "large":
        print("use large config")
        return configs.LargeConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


class MixDataInput(object):
    def __init__(self, data, batch_size=1, name=None):
        self.name = name
        self.batch_size = batch_size
        self.input_data = data

    def data(self):
        return self.input_data

    def next_batch(self):
        return self.input_data.next(self.batch_size)


class MixDataModel(object):
    def __init__(self, is_training, config, data):
        self._is_training = is_training
        self._config = config
        self._data = data
        data_width = matrix_data_width * matrix_data_height + flat_data_length + 1
        self._input_data = tf.placeholder(tf.float32, [None, data_width], name="input_data")
        x = tf.slice(self._input_data, [0, 0], [-1, matrix_data_width * matrix_data_height])
        x_f = tf.slice(self._input_data, [0, matrix_data_width * matrix_data_height], [-1, flat_data_length])
        y_ = tf.slice(self._input_data, [0, matrix_data_width * matrix_data_height + flat_data_length], [-1, 1])

        x_image = tf.reshape(x, [-1, matrix_data_height, matrix_data_width, 1])  # M * 30 * 5 * 1

        filter_sizes = self._config.filter_sizes
        flat_size = 0
        pooled_outputs = []
        for i, filter1_height in enumerate(filter_sizes):
            with tf.name_scope("conv" + str(i)):
                n_filters1 = 128

                W_conv1 = self.weight_variable([filter1_height, matrix_data_width, 1, n_filters1], "conv" + str(i))
                b_conv1 = self.bias_variable([n_filters1], "conv" + str(i))
                h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)  # M * 26 * 1 * 32
                conv1_height = matrix_data_height
                if self._config.padding_valid:
                    conv1_height = matrix_data_height - filter1_height + 1  # padding=VALID
                h_pool1 = self.max_pool_2x2(h_conv1)  # M * 13 * 1 * 32
                pooled_height = conv1_height // 2

                partial_flat_size = pooled_height * 1 * n_filters1
                h1_flat = tf.reshape(h_pool1, [-1, partial_flat_size])

                flat_size += partial_flat_size
                pooled_outputs.append(h1_flat)
        # end for

        if flat_data_length > 0:
            pooled_outputs.append(x_f)
            flat_size += flat_data_length
        h_flat = tf.concat(pooled_outputs, 1)

        nc_1_size = self._config.fc_size
        with tf.name_scope("fc1"):
            W_fc1 = self.weight_variable([flat_size, nc_1_size], "fc1")
            b_fc1 = self.bias_variable([nc_1_size], "fc1")
            h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

        self._keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self._keep_prob)

        with tf.name_scope("fc2"):
            W_fc2 = self.weight_variable([nc_1_size, 1], "fc2")
            b_fc2 = self.bias_variable([1], "fc2")
            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        self._cost = tf.reduce_mean(tf.pow(y_conv - y_, 2))
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
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            return tf.get_variable("b", shape, dtype=tf.float32)

    def conv2d(self, x, W):
        padding = 'VALID' if self._config.padding_valid == True else 'SAME'
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

    def max_pool_2x2(self, x):
        padding = 'VALID' if self._config.padding_valid == True else 'SAME'
        return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],
                              strides=[1, 2, 1, 1], padding=padding)

    @property
    def is_training(self):
        return self._is_training

    @property
    def data(self):
        return self._data

    @property
    def input_data(self):
        return self._input_data

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


def run_epoch(session, model, eval_op=None, iter=0, verbose=False):
    fetches = {
        "accuracy": model.accuracy,
        "buy_success": model.buy_success,
        "idea_target_buy": model.idea_target_buy,
        "b": model.b,
        "cost": model.cost
    }

    keep_prob = 1.0
    if eval_op is not None:
        fetches["eval_op"] = eval_op
        keep_prob = 0.5

    a_input_data = model.data.next_batch() if model.is_training else model.data.data()

    feed_dict = {model.keep_prob: keep_prob, model.input_data: a_input_data}

    vals = session.run(fetches, feed_dict)

    if verbose:
        accuracy = vals["accuracy"]
        buy_success = vals["buy_success"]
        cost = vals["cost"]
        print("step %d, accuracy %g, current cost: %g, buy_success: %g" % (
            iter, accuracy, cost, buy_success))

    return vals


config = get_config()
eval_config = get_config()
eval_config.batch_size = 1
eval_config.num_steps = 1

min = True

suffix = '.txt'
if (min):
    suffix = '_min.txt'
train_path = train_file + suffix
test_path = test_file + suffix

data = InputData(model_fields=model_fields, file_path=train_path)
test = InputData(model_fields=model_fields, file_path=test_path)
test_data = test.data()

if config.valid_on_test_data:
    train_data = data
    valid_data = test_data
else:
    train_data, valid = data.random_pick(0.7)
    valid_data = valid.data()

print("valid data count:" + str(len(valid_data)))


with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    with tf.name_scope("Train"):
        train_input = MixDataInput(data=train_data, batch_size=config.batch_size, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = MixDataModel(is_training=True, config=config, data=train_input)
        tf.summary.scalar("Training Loss", m.cost)

    with tf.name_scope("Valid"):
        valid_input = MixDataInput(data=valid_data, name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = MixDataModel(is_training=False, config=config, data=valid_input)
        tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
        test_input = MixDataInput(data=test_data, name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = MixDataModel(is_training=False, config=eval_config, data=test_input)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
        for i in range(config.num_epochs):

            verbose = True
            if i % 100 == 0:
                verbose = True
            else:
                verbose = False
            vals = run_epoch(session, m, eval_op=m.train_op, verbose=verbose, iter=i)

            if i % 1000 == 0:
                vals = run_epoch(session, mvalid)
                print("Epoch: %d Valid accuracy: %f, cost:%.3f, idea_target_buy:%.1f" % (
                i, vals["accuracy"], vals["cost"], vals["b"]))

            if i % 10000 == 0:
                vals = run_epoch(session, mtest)
                print("Test accuracy: %f" % vals["accuracy"])

        vals = run_epoch(session, mtest)
        print("Test accuracy: %f" % vals["accuracy"])

        if FLAGS.save_path:
            print("Saving model to %s." % FLAGS.save_path)
            sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)
