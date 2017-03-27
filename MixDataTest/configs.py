class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    keep_prob = 1.0
    batch_size = 50
    num_epochs = 5000
    beta = 0.1
    cnn_filters = [[1, 3, 32], [2, 5, 64]]  # for each item, [a,b,c] a num of convs in one layer, b conv size, c filter count
    fc_size = 128
    padding_valid = True
    valid_on_test_data = False


class MidConfig(object):
    """Medium config."""
    init_scale = 0.05
    keep_prob = 1.0
    batch_size = 100
    num_epochs = 50000
    beta = 0.1
    cnn_filters = [[2, 3, 128], [3, 5, 256]]  # for each item, [a,b,c] a num of convs in one layer, b conv size, c filter count
    fc_size = 256
    padding_valid = True
    valid_on_test_data = False


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    keep_prob = 1.0
    batch_size = 100
    num_epochs = 200000
    beta = 0.1
    cnn_filters = [[2, 3, 64], [3, 5, 128], [3, 11, 256], [3, 19, 256], [4, 31, 512]]  # for each item, [a,b,c] a num of convs in one layer, b conv size, c filter count
    fc_size = 1024
    padding_valid = True
    valid_on_test_data = False
