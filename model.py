import paddle.fluid as fluid


def crowd_deconv_without_bn(img):
    x = img
    x = fluid.layers.conv2d(input=x, num_filters=64, filter_size=3, padding=1, act='relu')
    x = fluid.layers.conv2d(input=x, num_filters=64, filter_size=3, padding=1, act='relu')
    x = fluid.layers.pool2d(input=x, pool_size=2, pool_type="max", pool_stride=2)
    x = fluid.layers.conv2d(input=x, num_filters=128, filter_size=3, padding=1, act='relu')
    x = fluid.layers.conv2d(input=x, num_filters=128, filter_size=3, padding=1, act='relu')
    x = fluid.layers.pool2d(input=x, pool_size=2, pool_type="max", pool_stride=2)
    x = fluid.layers.conv2d(input=x, num_filters=256, filter_size=3, padding=1, act='relu')
    x = fluid.layers.conv2d(input=x, num_filters=256, filter_size=3, padding=1, act='relu')
    x = fluid.layers.conv2d(input=x, num_filters=256, filter_size=3, padding=1, act='relu')
    x = fluid.layers.pool2d(input=x, pool_size=2, pool_type="max", pool_stride=2)
    x = fluid.layers.conv2d(input=x, num_filters=512, filter_size=3, padding=1, act='relu')
    x = fluid.layers.conv2d(input=x, num_filters=512, filter_size=3, padding=1, act='relu')
    x = fluid.layers.conv2d(input=x, num_filters=512, filter_size=3, padding=1, act='relu')
    x = fluid.layers.pool2d(input=x, pool_size=3, pool_type="max", pool_stride=1, pool_padding=1)
    x = fluid.layers.dropout(x=x, dropout_prob=0.5)
    x = fluid.layers.conv2d(input=x, num_filters=512, filter_size=3, padding=1, act='relu')
    x = fluid.layers.conv2d(input=x, num_filters=512, filter_size=3, padding=1, act='relu')
    x = fluid.layers.conv2d(input=x, num_filters=512, filter_size=3, padding=1)
    return x


def dilations_cnn(img):
    x = img
    x = fluid.layers.conv2d(input=x, num_filters=24, filter_size=5, padding=2, act='relu')
    x = fluid.layers.pool2d(input=x, pool_type='avg', pool_size=5, pool_stride=2, pool_padding=2)
    x = fluid.layers.conv2d(input=x, num_filters=24, filter_size=5, padding=2, act='relu')
    x = fluid.layers.pool2d(input=x, pool_type='avg', pool_size=5, pool_stride=2, pool_padding=2)
    x = fluid.layers.conv2d(input=x, num_filters=24, filter_size=5, padding=2, act='relu')
    x = fluid.layers.pool2d(input=x, pool_type='avg', pool_size=5, pool_stride=2, pool_padding=2)
    return x
