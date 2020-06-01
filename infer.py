import csv
import os
import numpy as np
import paddle.fluid as fluid
from PIL import Image
from model import crowd_deconv_without_bn, dilations_cnn

# 是否使用GPU
use_cuda = True
# 预测模型保存；路径
infer_model_path = 'infer_model/'
# 模型参数保存路径
persistables_model_path = 'persistables_model/'

images = fluid.layers.data(name='images', shape=[3, 640, 480], dtype='float32')

# 创建CrowdNet网络模型
net_out1 = crowd_deconv_without_bn(images)
net_out2 = dilations_cnn(images)
concat_out = fluid.layers.concat([net_out1, net_out2], axis=1)
concat_out = fluid.layers.batch_norm(input=concat_out, act='relu')
conv_end = fluid.layers.conv2d(input=concat_out, num_filters=1, filter_size=1)
map_out = fluid.layers.resize_bilinear(conv_end, out_shape=(80, 60))
sum_ = fluid.layers.reduce_sum(map_out, dim=[1, 2, 3])
sum_ = fluid.layers.reshape(sum_, [-1, 1])

# 创建
infer_program = fluid.default_main_program().clone(for_test=False)

# 定义执行器
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())


# 加载模型参数
def if_exist(var):
    if os.path.exists(os.path.join(persistables_model_path, var.name)):
        print('loaded: %s' % var.name)
    return os.path.exists(os.path.join(persistables_model_path, var.name))


fluid.io.load_vars(exe, persistables_model_path, main_program=infer_program, predicate=if_exist)


# 对图像进行预处理
def load_data(path):
    ig_cv_img = Image.open(path)
    test_img = ig_cv_img.resize((640, 480), Image.ANTIALIAS)
    test_im = np.array(test_img) / 255.0
    test_im = test_im.transpose().reshape(1, 3, 640, 480).astype('float32')
    return test_im


# 执行预测
def infer(path):
    test_im = load_data(path)
    q = exe.run(program=infer_program,
                feed={images.name: test_im},
                fetch_list=[sum_])[0][0][0]
    return int(q + 0.5)


if __name__ == '__main__':
    # 预测结果都放在这个字典中
    data_dict = {}
    test_path = 'data/test/'
    images1 = os.listdir(test_path)
    for image in images1:
        image_path = os.path.join(test_path, image)
        r = infer(image_path)
        print(r)
        data_dict[image[:-4]] = r

    with open('results.csv', 'w', newline='') as csvfile:
        fieldnames = ['id', 'predicted']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for k, v in data_dict.items():
            writer.writerow({'id': k, 'predicted': v})
