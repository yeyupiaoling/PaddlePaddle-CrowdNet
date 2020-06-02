import paddle.fluid as fluid
from paddle.fluid import ParamAttr
from paddle.fluid.initializer import MSRA


def conv_bn(input, num_filters, filter_size, padding, stride=1, num_groups=1, act='relu'):
    parameter_attr = ParamAttr(learning_rate=0.1, initializer=MSRA())
    conv = fluid.layers.conv2d(input=input,
                               num_filters=num_filters,
                               filter_size=filter_size,
                               stride=stride,
                               padding=padding,
                               groups=num_groups,
                               param_attr=parameter_attr,
                               bias_attr=False)
    return fluid.layers.batch_norm(input=conv, act=act)


def crowd_deconv_without_bn(img):
    x = img
    x = conv_bn(input=x, num_filters=64, filter_size=3, padding=1, act='relu')
    x = conv_bn(input=x, num_filters=64, filter_size=3, padding=1, act='relu')
    x = fluid.layers.pool2d(input=x, pool_size=2, pool_stride=2)
    x = conv_bn(input=x, num_filters=128, filter_size=3, padding=1, act='relu')
    x = conv_bn(input=x, num_filters=128, filter_size=3, padding=1, act='relu')
    x = fluid.layers.pool2d(input=x, pool_size=2, pool_stride=2)
    x = conv_bn(input=x, num_filters=256, filter_size=3, padding=1, act='relu')
    x = conv_bn(input=x, num_filters=256, filter_size=3, padding=1, act='relu')
    x = conv_bn(input=x, num_filters=256, filter_size=3, padding=1, act='relu')
    x = fluid.layers.pool2d(input=x, pool_size=2, pool_stride=2)
    x = conv_bn(input=x, num_filters=512, filter_size=3, padding=1, act='relu')
    x = conv_bn(input=x, num_filters=512, filter_size=3, padding=1, act='relu')
    x = conv_bn(input=x, num_filters=512, filter_size=3, padding=1, act='relu')
    x = fluid.layers.pool2d(input=x, pool_size=3, pool_stride=1, pool_padding=1)
    x = conv_bn(input=x, num_filters=512, filter_size=3, padding=1, act='relu')
    x = conv_bn(input=x, num_filters=512, filter_size=3, padding=1, act='relu')
    x = conv_bn(input=x, num_filters=512, filter_size=3, padding=1)
    return x


def dilations_cnn(img):
    x = img
    x = conv_bn(input=x, num_filters=24, filter_size=5, padding=2, act='relu')
    x = fluid.layers.pool2d(input=x, pool_size=2, pool_type='avg', pool_stride=2)
    x = conv_bn(input=x, num_filters=24, filter_size=5, padding=2, act='relu')
    x = fluid.layers.pool2d(input=x, pool_size=2, pool_type='avg', pool_stride=2)
    x = conv_bn(input=x, num_filters=24, filter_size=5, padding=2, act='relu')
    x = fluid.layers.pool2d(input=x, pool_size=2, pool_type='avg', pool_stride=2)
    return x