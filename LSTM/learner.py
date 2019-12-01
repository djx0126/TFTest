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
total_train_samples=60000
total_test_samples=10000
lr_decay_epochs=1
output_folder="./model_output"
#用来保存模型以及我们需要的所有东西
if not os.path.exists(output_folder):
    os.makedirs(output_folder)



save_format="hdf5" #或saved_model
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

file_path = "./data/raw_data_pre64_gain5_20130415_20191025_small.txt"
meta_file_path = "./data/raw_data_pre64_gain5_20130415_20191025_meta.txt"

df, gain_coded = loader.prepare_data_frame(file_path)
M = len(df)

values = tf.reshape(df.values, [-1, 5, 64])
gain = keras.utils.to_categorical(gain_coded.values,num_classes=3)

splitpoint = int(round(M * 0.8))
splitpoint = int(np.floor(splitpoint/batch_size) * batch_size)
(X_train, X_val) = (df.values[0:splitpoint], df.values[splitpoint:])
(Y_train, Y_val) = (gain[0:splitpoint], gain[splitpoint:])

train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train.shuffle(Y_train.shape[0]).batch(batch_size)

val = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
val_dataset = val.shuffle(Y_val.shape[0]).batch(2)

train_ds=train_dataset.shuffle(buffer_size=batch_size*10).batch(batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE).repeat()
train_steps_per_epoch = np.floor(len(X_train)/batch_size).astype(np.int32)


def get_compiled_model():
    model = tf.keras.Sequential([
        # tf.keras.layers.Dense(1024, activation='relu'),
        # tf.keras.layers.Dense(1024, activation='relu'),
        # tf.keras.layers.Conv1D(1, 1, strides=1, padding='valid', data_format='channels_first'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'categorical_accuracy'])

    return model


model = get_compiled_model()

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
# csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(log_dir,'logs.log'),separator=',')
# keras.callbacks.ModelCheckpoint(output_model_file, save_best_only = True)
# callbacks=[ckpt,earlystop,lr,tensorboard,terminate,reduce_lr,csv_logger]
callbacks = [tensorboard]

H = model.fit(train_ds, epochs=20, verbose=2, callbacks=callbacks, validation_data=val_dataset,
              steps_per_epoch=train_steps_per_epoch, validation_steps=np.floor(train_steps_per_epoch/10))
print(H.history['accuracy'])
print(H.history['loss'])
print(H.history)

print("The model architure:\n")
print(model.summary())

ploter.plot_save(H, os.path.join(log_dir, profile_dir_name))


