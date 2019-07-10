import os
import shutil
import numpy as np
import paddle
import paddle.fluid as fluid

from model import crowd_deconv_without_bn, dilations_cnn
from myreader import train_set

np.set_printoptions(threshold=np.inf)


images = fluid.layers.data(name='images', shape=[3, 640, 480], dtype='float32')
label = fluid.layers.data(name='label', shape=[1, 80, 60], dtype='float32')
img_num = fluid.layers.data(name='img_num', shape=[1], dtype='float32')
net_out1 = crowd_deconv_without_bn(images)
net_out2 = dilations_cnn(images)

concat_out = fluid.layers.concat([net_out1, net_out2], axis=1)
concat_out = fluid.layers.batch_norm(input=concat_out, act='relu')
conv_end = fluid.layers.conv2d(input=concat_out, num_filters=1, filter_size=1)
map_out = fluid.layers.resize_bilinear(conv_end, out_shape=(80, 60))
sum_ = fluid.layers.reduce_sum(map_out, dim=[1, 2, 3])
sum_ = fluid.layers.reshape(sum_, [-1, 1])

cost = fluid.layers.square_error_cost(input=map_out, label=label)
avg_cost = fluid.layers.mean(cost)

loss_ = fluid.layers.abs(fluid.layers.elementwise_sub(sum_, img_num))
loss = fluid.layers.mean(fluid.layers.elementwise_div(loss_, img_num))
sum_loss = loss + avg_cost * 6e5

optimizer = fluid.optimizer.Adam(learning_rate=0.0001)
optimizer.minimize(sum_loss)

# 设置训练场所
use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

feeder = fluid.DataFeeder(feed_list=[images, label, img_num], place=place)

# 设置训练reader
train_reader = paddle.batch(reader=train_set(), batch_size=2)

# 开始训练
for epochs in range(20):
    for batch_id, train_data in enumerate(train_reader()):
        train_cost, sult, lab, predict_sum, label_sum = exe.run(program=fluid.default_main_program(),
                                                                feed=feeder.feed(train_data),
                                                                fetch_list=[sum_loss, map_out, label, sum_, img_num])

        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, predict:%0.5f, label:%0.5f, predict_sum:%0.5f, label_sum:%0.5f' % (
                epochs, batch_id, train_cost[0], np.sum(sult), np.sum(lab), predict_sum[0], label_sum[0]))

    # 保存模型
    model_save_dir = 'save_model/%d' % epochs
    shutil.rmtree(model_save_dir, ignore_errors=True)
    os.makedirs(model_save_dir)
    fluid.io.save_inference_model(model_save_dir, ['images'], [sum_], exe)
