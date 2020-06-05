import os
import shutil
import paddle
import paddle.fluid as fluid
from model import deep_network, shallow_network
import reader


# 是否使用GPU
USE_CUDA = True
# 模型参数保存路径
PERSISTABLES_MODEL_PATH = 'persistables_model/'
# 预测模型保存路径
INFER_MODEL = 'infer_model/'
# 训练轮数
EPOCHS_SUM = 400
# Batch大小
BATCH_SIZE = 6

# 定义输入层
images = fluid.data(name='images', shape=[None, 3, 640, 480], dtype='float32')
label = fluid.data(name='label', shape=[None, 1, 80, 60], dtype='float32')
img_num = fluid.data(name='img_num', shape=[None, 1], dtype='float32')

# 创建CrowdNet网络模型
net_out1 = deep_network(images)
net_out2 = shallow_network(images)
concat_out = fluid.layers.concat([net_out1, net_out2], axis=1)
conv_end = fluid.layers.conv2d(input=concat_out, num_filters=1, filter_size=1)
# 双向性插值
map_out = fluid.layers.resize_bilinear(conv_end, out_shape=(80, 60))
# 避开Batch维度求和
sum_ = fluid.layers.reduce_sum(map_out, dim=[1, 2, 3])
sum_ = fluid.layers.reshape(sum_, [-1, 1])

# 定义损失函数
loss = fluid.layers.square_error_cost(input=map_out, label=label) * 6e5
loss = fluid.layers.mean(loss)

optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(loss)

# 定义执行器
place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 定义异步数据读取
py_reader = fluid.io.PyReader(feed_list=[images, label, img_num],
                              capacity=32,
                              iterable=True,
                              return_list=False)
py_reader.decorate_sample_list_generator(paddle.batch(reader.train_reader2(), batch_size=BATCH_SIZE),
                                         places=fluid.core.CPUPlace())

# 加载训练模型
if PERSISTABLES_MODEL_PATH is not None and os.path.exists(PERSISTABLES_MODEL_PATH):
    def if_exist(var):
        if os.path.exists(os.path.join(PERSISTABLES_MODEL_PATH, var.name)):
            print('loaded: %s' % var.name)
        return os.path.exists(os.path.join(PERSISTABLES_MODEL_PATH, var.name))


    fluid.io.load_vars(exe, PERSISTABLES_MODEL_PATH, main_program=fluid.default_main_program(), predicate=if_exist)

# 开始训练
for epochs in range(EPOCHS_SUM):
    for batch_id, train_data in enumerate(py_reader()):
        train_loss, lab, predict_sum, label_sum = exe.run(program=fluid.default_main_program(),
                                                          feed=train_data,
                                                          fetch_list=[loss, label, sum_, img_num])

        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, predict_sum:%0.2f, label_sum:%0.2f' % (
                epochs, batch_id, train_loss[0], predict_sum[0], label_sum[0]))

    # 保存模型
    shutil.rmtree(PERSISTABLES_MODEL_PATH, ignore_errors=True)
    os.makedirs(PERSISTABLES_MODEL_PATH)
    fluid.io.save_persistables(exe, PERSISTABLES_MODEL_PATH)

    # 保存预测模型
    shutil.rmtree(INFER_MODEL, ignore_errors=True)
    os.makedirs(INFER_MODEL)
    fluid.io.save_inference_model(INFER_MODEL, [images.name], [map_out, sum_], exe)
