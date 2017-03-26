import tensorflow as tf

from input_data import InputData
from model_fields import ModelFields
import configs
from mix_data_model import MixDataModel, MixDataInput

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

    feed_dict = {model.keep_prob: keep_prob, model.mixed_data: a_input_data}

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
test_data = test

if config.valid_on_test_data:
    train_data = data
    valid_data = test_data
else:
    train_data, valid = data.random_pick(0.7)
    valid_data = valid

print("valid data count:" + str(len(valid_data.data())))


with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    with tf.name_scope("Train"):
        train_input = MixDataInput(data=train_data, batch_size=config.batch_size, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = MixDataModel(is_training=True, config=config, data_input=train_input)
        tf.summary.scalar("Training Loss", m.cost)

    with tf.name_scope("Valid"):
        valid_input = MixDataInput(data=valid_data, name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = MixDataModel(is_training=False, config=config, data_input=valid_input)
        tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
        test_input = MixDataInput(data=test_data, name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = MixDataModel(is_training=False, config=eval_config, data_input=test_input)

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
