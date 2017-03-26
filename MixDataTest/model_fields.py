import numpy as np

class ModelFields:
    def __init__(self, day_fields, ma_fields, over_all_ma_fields):
        self.day_fields = day_fields
        self.field_depth = sum(day_fields)
        self.ma_fields = ma_fields
        self.ma_size = len(ma_fields)
        self.over_all_ma_fields = over_all_ma_fields
        self.over_all_ma_size = len(over_all_ma_fields)
        self.size_of_one_data = self.field_depth + self.ma_size + self.over_all_ma_size + 1 # gain as percentage

    def one_data_size(self):
        return self.size_of_one_data

    def get_day_fields_by_day(self):
        seq = np.arange(self.size_of_one_data)
        field_type_count = len(self.day_fields)  # how many types of day_fields
        max_pre_days = max(self.day_fields) # max pre day_fields count

        index = 0
        output = np.zeros((field_type_count, max_pre_days))
        for i in np.arange(field_type_count):
            day_fields = self.day_fields[i]
            for j in np.arange(day_fields):
                output[i,j] = seq[index] + 1
                index+=1

        output = np.transpose(output)
        output = output.reshape((1 , -1))
        output = output[output > 0] - 1
        return output

    def get_ma_field_columns(self):
        seq = np.arange(self.size_of_one_data)
        seq_start = self.field_depth
        seq_end = self.field_depth + self.ma_size
        return seq[seq_start: seq_end]

    def get_over_all_ma_field_columns(self):
        seq = np.arange(self.size_of_one_data)
        seq_start = self.field_depth + self.ma_size
        seq_end = self.field_depth + self.ma_size + self.over_all_ma_size
        return seq[seq_start: seq_end]

    def get_label_columns(self):
        seq = np.arange(self.size_of_one_data)
        seq_start = self.field_depth + self.ma_size + self.over_all_ma_size
        seq_end = self.size_of_one_data
        return seq[seq_start: seq_end]

    @property
    def matrix_data_width(self):
        return np.count_nonzero(self.day_fields)


    @property
    def matrix_data_height(self):
        return np.max(self.day_fields)

    @property
    def flat_data_length(self):
        return self.ma_size + self.over_all_ma_size

if __name__ == '__main__':
    model_fields = ModelFields(day_fields=[3,0,3,15,1], ma_fields=[5, 10, 20, 30, 60], over_all_ma_fields=[5, 10, 20, 30, 60])
    print(model_fields.one_data_size())
    print('day_field_cols: ' + str(model_fields.get_day_fields_by_day()))
    print('ma_field_cols: ' + str(model_fields.get_ma_field_columns()))
    print('over_all_ma_field_cols: ' + str(model_fields.get_over_all_ma_field_columns()))
    print('matrix_data_width: ' + str(model_fields.matrix_data_width))
    print('matrix_data_height: ' + str(model_fields.matrix_data_height))
    print('flat_data_length: ' + str(model_fields.flat_data_length))
