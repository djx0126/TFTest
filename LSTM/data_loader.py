import tensorflow as tf;
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

physical_devices = tf.config.experimental.list_physical_devices('GPU')#列出所有可见显卡
print("All the available GPUs:\n",physical_devices)

np.set_printoptions(precision=5, suppress=True)

file_path = "./data/raw_data_pre64_gain5_20130415_20191025_small.txt"
meta_file_path = "./data/raw_data_pre64_gain5_20130415_20191025_meta.txt"


def load_meta(file_path):
    print('reading from file ', file_path)
    with open(file_path, 'r') as test_data:
        all_lines = test_data.readlines()
        m = len(all_lines)
        meta = {}

        index = 0
        for line in all_lines:
            if line.startswith("pre="):
                meta['pre'] = int(line.replace("pre=", ""))
            if line.startswith("gain="):
                meta['gain'] = int(line.replace("gain=", ""))
            if line.startswith("dayFields="):
                meta['dayFields'] = line.replace("dayFields=", "").strip()
                strs = meta['dayFields'].split(",")
                fields = ["c", "o", "h", "l", "v"]
                idx = 0
                for str in strs:
                    meta[fields[idx]] = int(str)
                    idx += 1
            index += 1
        return meta


def build_column_label(meta):
    columns = ['date', 'gain', 'code']
    fields = ['c', 'o', 'h', 'l', 'v']
    for f in fields:
        l = meta[f]
        for n in range(1, l + 1):
            columns.append(f + "" + str(n))
    return columns


meta = load_meta(meta_file_path)
print(meta)
column_label = build_column_label(meta)
# print(column_label)

def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=4, # 为了示例更容易展示，手动设置较小的值
        label_name='gain',
        na_value="?",
        num_epochs=1,
        ignore_errors=True)
    return dataset


# raw_train_data = get_dataset(file_path)
# examples, labels = next(iter(raw_train_data)) # 第一个批次
# print("EXAMPLES: \n", examples, "\n")
# print("LABELS: \n", labels)

def prepare_data_frame(file_path):
    df = pd.read_csv(file_path, sep=',')

    gains = df['gain']
    sorted_gain_value = gains.sort_values().to_numpy()
    m = len(sorted_gain_value)
    part = m//3
    div1 = sorted_gain_value[part]
    div2 = sorted_gain_value[part + part - 1]
    print('gain segments: ', sorted_gain_value[0], sorted_gain_value[part], sorted_gain_value[part + part - 1], sorted_gain_value[m - 1])

    df.loc[df.gain < div1, 'gain_coded'] = int(0)
    df.loc[(df.gain >= div1) & (df.gain < div2), 'gain_coded'] = int(1)
    df.loc[df.gain >= div2, 'gain_coded'] = int(2)
    df['gain_coded'] = pd.Categorical(df['gain_coded'])
    df['gain_coded'] = df.gain_coded.cat.codes

    df.drop(['date'], axis=1, inplace=True)
    df.drop(['code'], axis=1, inplace=True)
    df.drop(['gain'], axis=1, inplace=True)

    print(df.head(5))

    df = df.sample(frac=1)
    print(df.head(5))

    gain_coded = df.pop('gain_coded')
    return df, gain_coded





def process_continuous_data(mean, data):
    # 标准化数据
    data = tf.cast(data, tf.float32) * 1/(2*mean)
    return tf.reshape(data, [-1, 1])


if __name__ == '__main__':
    print("done")
