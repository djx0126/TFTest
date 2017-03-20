#!user/bin/env python3
# -*- coding: gbk -*-

import numpy as np
import re
from model_fields import ModelFields

class InputData(object):
    def __init__(self, model_fields, file_path=None, src_data=None):
        self.model_fields = model_fields
        self.one_data_size = self.model_fields.one_data_size()
        self.__file_path = file_path
        self.M = 0
        self.__raw_data = []
        self.__seq = []
        self.__next_batch_index = 0
        if (src_data is not None):
            self.__file_path = None
            self.__raw_data = src_data
            self.M = len(src_data)
            self.__seq = np.arange(self.M)
            self.suffle()
        else:
            self.read_data()


    def size(self):
        return self.M

    def read_data(self):
        print('reading from file ', self.__file_path)
        with open(self.__file_path, 'r') as test_data:
            all_lines = test_data.readlines()
            m = len(all_lines)

            index = 0
            return_mat = np.zeros((m, self.one_data_size))
            for line in all_lines:
                arr = re.split(r'\s+', line.strip())
                return_mat[index, :] = arr
                if (len(arr) != self.one_data_size):
                    print(arr)
                index += 1

            self.__raw_data = return_mat
            self.M = m
            self.__seq = np.arange(self.M)
            self.suffle()

    def suffle(self):
        np.random.shuffle(self.__seq)
        self.__next_batch_index = 0

    def next(self, batch_size):
        batch_start = self.__next_batch_index
        batch_end = batch_start + batch_size
        if (batch_end > self.M):
            self.suffle()
            return self.next(batch_size)
        seq = self.__seq[batch_start:batch_end]
        batch = self.__raw_data[seq,:]
        self.__next_batch_index += batch_size
        return batch

    def random_pick(self, partial):
        if (partial > 0 and partial < 1):
            partial = int(partial * self.M)
            seq1 = self.__seq[0: partial]
            part1_data = self.__raw_data[seq1, :]
            part1 = InputData(model_fields=self.model_fields, src_data=part1_data)
            seq2 = self.__seq[partial + 1: self.M]
            part2_data = self.__raw_data[seq2, :]
            part2 = InputData(model_fields=self.model_fields, src_data=part2_data)
            return (part1, part2)
        return self

    def get_day_field_cols(self):
        cols = self.model_fields.get_day_fields_by_day()
        return self.__raw_data[self.__seq, cols]

    def get_ma_fields_cols(self):
        cols = self.model_fields.get_ma_field_columns()
        return self.__raw_data[self.__seq, cols]

    def get_over_all_ma_fields_cols(self):
        cols = self.model_fields.get_over_all_ma_field_columns()
        return self.__raw_data[self.__seq, cols]

    def get_label_cols(self):
        cols = self.model_fields.get_label_columns()
        return self.__raw_data[self.__seq, cols]


if __name__ == '__main__':
    model_fields = ModelFields(day_fields=[30,30,30,30,30], ma_fields=[5, 10, 20, 30, 60], over_all_ma_fields=[5, 10, 20, 30, 60])
    input_data = InputData(model_fields=model_fields, file_path='test_data_min.txt')
    print(input_data.size())
    batch_data = input_data.next(3)
    for idx in range(3):
        print(batch_data[idx])

    labels = input_data.get_label_cols()
    print('3rd label: ' + str(labels[2]))

    train, test = input_data.random_pick(0.7)
    print('train size: ' + str(train.size()))
    print('test size: ' + str(test.size()))

