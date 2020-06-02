import os
import shutil
import paddle
import paddle.fluid as fluid
from model import crowd_deconv_without_bn, dilations_cnn
import reader

# 是否使用GPU
USE_GPU = True
# 模型参数保存路径
MODEL_PATH = 'models/'
# 训练轮数
EPOCHS_SUM = 400
# Batch大小
BATCH_SIZE = 6

# 定义输入层
images = fluid.data(name='images', shape=[None, 3, 640, 480], dtype='float32')
label = fluid.data(name='label', shape=[None, 1, 80, 60], dtype='float32')
img_num = fluid.data(name='img_num', shape=[None, 1], dtype='float32')

# 创建CrowdNet网络模型
net_out1 = crowd_deconv_without_bn(images)
net_out2 = dilations_cnn(images)
concat_out = fluid.layers.concat([net_out1, net_out2], axis=1)
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
place = fluid.CUDAPlace(0) if USE_GPU else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 加载训练模型
if MODEL_PATH is not None and os.path.exists(MODEL_PATH):
    def if_exist(var):
        if os.path.exists(os.path.join(MODEL_PATH, var.name)):
            print('loaded: %s' % var.name)
        return os.path.exists(os.path.join(MODEL_PATH, var.name))


    fluid.io.load_vars(exe, MODEL_PATH, main_program=fluid.default_main_program(), predicate=if_exist)

# 定义异步数据读取
py_reader = fluid.io.PyReader(feed_list=[images, label, img_num],
                              capacity=32,
                              iterable=True,
                              return_list=False)
py_reader.decorate_sample_list_generator(paddle.batch(reader.train_reader2(), batch_size=BATCH_SIZE),
                                         places=fluid.core.CPUPlace())

# 开始训练
for epochs in range(EPOCHS_SUM):
    for batch_id, train_data in enumerate(py_reader()):
        train_cost, lab, predict_sum, label_sum = exe.run(program=fluid.default_main_program(),
                                                          feed=train_data,
                                                          fetch_list=[sum_loss, label, sum_, img_num])

        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, predict_sum:%0.2f, label_sum:%0.2f' % (
                epochs, batch_id, train_cost[0], predict_sum[0], label_sum[0]))

    # 保存模型
    shutil.rmtree(MODEL_PATH, ignore_errors=True)
    os.makedirs(MODEL_PATH)
    fluid.io.save_persistables(exe, MODEL_PATH)
