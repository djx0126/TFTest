import os;
import datetime;
import tensorflow as tf;
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import keras;
import data_loader as loader;
import ploter;

batch_size=32
epochs=5
regularizer=1e-3
lr_decay_epochs=1


def prepare_folders():
    output_folder="./model_output"
    #用来保存模型以及我们需要的所有东西
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    save_format="hdf5"  # 或saved_model
    if save_format=="hdf5":
        save_path_models=os.path.join(output_folder,"hdf5_models")
        if not os.path.exists(save_path_models):
            os.makedirs(save_path_models)
        save_path=os.path.join(save_path_models,"ckpt_epoch{epoch:02d}_val_acc{val_accuracy:.2f}.hdf5")
    elif save_format=="saved_model":
        save_path_models=os.path.join(output_folder,"saved_models")
        if not os.path.exists(save_path_models):
            os.makedirs(save_path_models)
        save_path=os.path.join(save_path_models,"ckpt_epoch{epoch:02d}_val_acc{val_accuracy:.2f}.ckpt")
    #用来保存日志
    now = datetime.datetime.now()
    logs_dir_name = 'logs_{}'.format(now.strftime("%Y%m%d-%H%M%S"))
    profile_dir_name = '{}'.format(now.strftime("%Y-%m-%d_%H-%M-%S"))

    log_dir = os.path.join(output_folder, logs_dir_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    profile_log_dir = os.path.join(output_folder, logs_dir_name, 'train', 'plugins', 'profile', profile_dir_name)
    if not os.path.exists(profile_log_dir):
        os.makedirs(profile_log_dir)
    profile_log_dir = os.path.join(output_folder, 'train', 'plugins', 'profile', profile_dir_name)
    if not os.path.exists(profile_log_dir):
        os.makedirs(profile_log_dir)

    return log_dir, profile_dir_name


log_dir, profile_dir_name = prepare_folders()

file_path = "./data/raw_data_pre64_gain5_20130415_20191025_S.txt"
meta_file_path = "./data/raw_data_pre64_gain5_20130415_20191025_S_meta.txt"

test_by_date=20190101
# test_by_date=0

df = loader.prepare_data_frame(file_path)
print(df.head(5))
df_train = df

if test_by_date:
    df_train = df.loc[df.date < test_by_date]
    df_test = df.loc[df.date >= test_by_date]
    X_test = df_test.drop(['date'], axis=1, inplace=False)
    X_test = X_test.drop(['code'], axis=1, inplace=False)
    X_test = X_test.drop(['gain'], axis=1, inplace=False)
    gain_coded_test = X_test.pop('gain_coded')
    X_test = X_test.values
    Y_test = keras.utils.to_categorical(gain_coded_test.values, num_classes=3)


df_train = df_train.sample(frac=1)
dft = df_train.drop(['date'], axis=1, inplace=False)
dft = dft.drop(['code'], axis=1, inplace=False)
dft = dft.drop(['gain'], axis=1, inplace=False)
print(dft.head(5))

gain_coded = dft.pop('gain_coded')
gain = keras.utils.to_categorical(gain_coded.values, num_classes=3)

M = len(dft)
splitpoint = int(round(M * 0.8))
splitpoint = int(np.floor(splitpoint/batch_size) * batch_size)
(X_train, X_val) = (dft.values[0:splitpoint], dft.values[splitpoint:])
(Y_train, Y_val) = (gain[0:splitpoint], gain[splitpoint:])

train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train.shuffle(Y_train.shape[0]).batch(batch_size)

val = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
val_dataset = val.shuffle(Y_val.shape[0]).batch(batch_size)

# train_ds=train_dataset.shuffle(buffer_size=batch_size*10).batch(batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE).repeat()
train_ds=train_dataset.shuffle(buffer_size=batch_size*10)
train_ds=train
train_steps_per_epoch = np.floor(len(X_train)/batch_size).astype(np.int32)


def get_compiled_model():
    inputs = keras.Input(shape=(320,))
    x = keras.layers.Dense(8, activation='relu')(inputs)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(8, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(3, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
    model.summary()
    # keras.utils.plot_model(model, 'my_first_model_with_shape_info.png', show_shapes=True)

    # model = tf.keras.Sequential([
    #     # tf.keras.layers.Dense(1024, activation='relu'),
    #     # tf.keras.layers.Dense(1024, activation='relu'),
    #     # tf.keras.layers.Conv1D(1, 1, strides=1, padding='valid', data_format='channels_first'),
    #     tf.keras.layers.Dense(8, activation='relu'),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(8, activation='relu'),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(3, activation='softmax')
    # ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'categorical_accuracy'])

    return model


model = get_compiled_model()

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(log_dir,'logs.log'),separator=',')
# keras.callbacks.ModelCheckpoint(output_model_file, save_best_only = True)
# callbacks=[ckpt,earlystop,lr,tensorboard,terminate,reduce_lr,csv_logger]
# callbacks = [tensorboard]

with tf.device("/cpu:0"):
    # H = model.fit(train_ds, epochs=20, verbose=2, callbacks=callbacks, validation_data=val_dataset,
    H = model.fit(X_train, Y_train, epochs=10, verbose=2, validation_split=0.2, workers=12, #validation_data=val_dataset,
              steps_per_epoch=train_steps_per_epoch, validation_steps=int(np.floor(train_steps_per_epoch/5)))
    print(H.history['accuracy'])
    print(H.history['loss'])
    print(H.history)

    print("The model architecture:\n")
    print(model.summary())

    ploter.plot_save(H, os.path.join(log_dir, profile_dir_name))


def evaluate(X, Y, model, data_frame):
    ypref_val = model.predict(x=X, workers=2)
    # print(ypref_val)
    ypre_val = np.argmax(ypref_val, axis=1)
    sum_pre2 = np.sum(ypre_val == 2)
    sum_Y_2 = np.sum(np.argmax(Y[ypre_val == 2], axis=1) == 2)

    to_buy = data_frame[ypre_val == 2]
    # print(data_frame[ypre_val == 2])
    print('gain=', np.sum(to_buy['gain']))
    print('count=', sum_pre2, sum_Y_2, ' acc=', sum_Y_2/sum_pre2)
    threshold = 0.75
    count_thre = np.sum(ypref_val[:, 2] > threshold)
    acc_thre = np.sum(np.argmax(Y[ypref_val[:, 2] > threshold], axis=1) == 2) / np.sum(ypref_val[:, 2] > threshold)
    to_buy_thre = data_frame[ypref_val[:, 2] > threshold]
    print('buy with threshold ', threshold, ' count=', count_thre, ' acc=', acc_thre, 'gain=', np.sum(to_buy_thre['gain']))


evaluate(X_val, Y_val, model, df_train[splitpoint:])

if test_by_date:
    evaluate(X_test, Y_test, model, df_test)


