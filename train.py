import os
import shutil
import paddle
import paddle.fluid as fluid
from model import crowd_deconv_without_bn, dilations_cnn
import reader

# 是否使用GPU
use_cuda = True
# 模型参数保存路径
model_path = 'models/'
# 训练轮数
epochs_sum = 400
# Batch大小
batch_size = 6

# 定义输入层
images = fluid.layers.data(name='images', shape=[3, 640, 480], dtype='float32')
label = fluid.layers.data(name='label', shape=[1, 80, 60], dtype='float32')
img_num = fluid.layers.data(name='img_num', shape=[1], dtype='float32')

# 创建CrowdNet网络模型
net_out1 = crowd_deconv_without_bn(images)
net_out2 = dilations_cnn(images)
concat_out = fluid.layers.concat([net_out1, net_out2], axis=1)
concat_out = fluid.layers.batch_norm(input=concat_out, act='relu')
conv_end = fluid.layers.conv2d(input=concat_out, num_filters=1, filter_size=1)
map_out = fluid.layers.resize_bilinear(conv_end, out_shape=(80, 60))
sum_ = fluid.layers.reduce_sum(map_out, dim=[1, 2, 3])
sum_ = fluid.layers.reshape(sum_, [-1, 1])

# 定义损失函数
cost = fluid.layers.square_error_cost(input=map_out, label=label)
avg_cost = fluid.layers.mean(cost)
loss_ = fluid.layers.abs(fluid.layers.elementwise_sub(sum_, img_num))
loss = fluid.layers.mean(fluid.layers.elementwise_div(loss_, img_num))
sum_loss = loss + avg_cost * 6e5

optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(sum_loss)

# 定义执行器
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

feeder = fluid.DataFeeder(feed_list=[images, label, img_num], place=place)

# 定义reader
train_reader = fluid.io.shuffle(paddle.batch(reader=reader.train_reader(), batch_size=batch_size), buf_size=5000)

# 加载训练模型
if model_path is not None and os.path.exists(model_path):
    def if_exist(var):
        if os.path.exists(os.path.join(model_path, var.name)):
            print('loaded: %s' % var.name)
        return os.path.exists(os.path.join(model_path, var.name))


    fluid.io.load_vars(exe, model_path, main_program=fluid.default_main_program(), predicate=if_exist)

# 开始训练
for epochs in range(epochs_sum):
    for batch_id, train_data in enumerate(train_reader()):
        train_cost, lab, predict_sum, label_sum = exe.run(program=fluid.default_main_program(),
                                                          feed=feeder.feed(train_data),
                                                          fetch_list=[sum_loss, label, sum_, img_num])

        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, predict_sum:%0.2f, label_sum:%0.2f' % (
                epochs, batch_id, train_cost[0], predict_sum[0], label_sum[0]))

    # 保存模型
    shutil.rmtree(model_path, ignore_errors=True)
    os.makedirs(model_path)
    fluid.io.save_persistables(exe, model_path)
