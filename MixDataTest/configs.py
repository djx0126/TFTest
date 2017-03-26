class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    keep_prob = 1.0
    batch_size = 50
    num_epochs = 5000
    n_filters = [128]
    filter_sizes = [3, 5]
    fc_size = 128
    padding_valid = True
    valid_on_test_data = False


class MidConfig(object):
    """Medium config."""
    init_scale = 0.05
    keep_prob = 1.0
    batch_size = 100
    num_epochs = 50000
    n_filters = [256, 256]
    filter_sizes = [3, 5, 11]
    fc_size = 256
    padding_valid = True
    valid_on_test_data = False


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    keep_prob = 1.0
    batch_size = 100
    num_epochs = 200000
    n_filters = [256, 256]  # num_filters for each size of filter, each item will be one conv layer
    filter_sizes = [3, 5, 11, 19, 31]
    fc_size = 1024
    padding_valid = True
    valid_on_test_data = True
