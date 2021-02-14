# 前言
CrowdNet模型是2016年提出的人流密度估计模型，论文为《CrowdNet: A Deep Convolutional Network for DenseCrowd Counting》，CrowdNet模型主要有深层卷积神经网络和浅层卷积神经组成，通过输入原始图像和高斯滤波器得到的密度图进行训练，最终得到的模型估计图像中的行人的数量。当然这不仅仅可以用于人流密度估计，理论上其他的动物等等的密度估计应该也可以。

本项目开发环境为：
 - Windows 10
 - Python 3.7
 - PaddlePaddle 2.0.0a0

# CrowdNet模型实现
以下是CrowdNet模型的结构图，从结构图中可以看出，CrowdNet模型是深层卷积网络（Deep Network）和浅层卷积网络（Shallow Network）组成，两组网络通过拼接成一个网络，接着输入到一个卷积核数量和大小都是1的卷积层，最后通过插值方式得到一个密度图数据，通过统计这个密度就可以得到估计人数

![CrowdNet模型结构](https://s1.ax1x.com/2020/06/04/tw4No6.jpg)

在PaddlePaddle中，通过以下代码判断即可实现上面的CrowdNet模型，在深层卷积网络和浅层卷积网络的卷积层都使用conv_bn卷积层，这个是通过把卷积层和batch_norm组合在一起的。在本项目中，输入的图像大小[3, 640, 480]，密度图大小为[1, 80, 60]，所以深层卷积网络输出的shape为[512, 80, 60]，浅层神经网络的输出为[24, 80, 60]。两个网络的输出通过`fluid.layers.concat()`接口进行拼接，拼接后输入到`fluid.layers.conv2d()`，最后通过`fluid.layers.resize_bilinear()` 双向性插值法输出一个密度图，最后使用的`fluid.layers.reduce_sum()`是为了方便在预测时直接输出估计人数。
```python
def deep_network(img):
    x = img
    x = conv_bn(input=x, num_filters=64, filter_size=3, padding=1, act='relu')
    x = conv_bn(input=x, num_filters=64, filter_size=3, padding=1, act='relu')
    x = fluid.layers.pool2d(input=x, pool_size=2, pool_stride=2)
    x = fluid.layers.dropout(x=x, dropout_prob=0.25)
    x = conv_bn(input=x, num_filters=128, filter_size=3, padding=1, act='relu')
    x = conv_bn(input=x, num_filters=128, filter_size=3, padding=1, act='relu')
    x = fluid.layers.pool2d(input=x, pool_size=2, pool_stride=2)
    x = fluid.layers.dropout(x=x, dropout_prob=0.25)
    x = conv_bn(input=x, num_filters=256, filter_size=3, padding=1, act='relu')
    x = conv_bn(input=x, num_filters=256, filter_size=3, padding=1, act='relu')
    x = conv_bn(input=x, num_filters=256, filter_size=3, padding=1, act='relu')
    x = fluid.layers.pool2d(input=x, pool_size=2, pool_stride=2)
    x = fluid.layers.dropout(x=x, dropout_prob=0.5)
    x = conv_bn(input=x, num_filters=512, filter_size=3, padding=1, act='relu')
    x = conv_bn(input=x, num_filters=512, filter_size=3, padding=1, act='relu')
    x = conv_bn(input=x, num_filters=512, filter_size=3, padding=1, act='relu')
    x = fluid.layers.pool2d(input=x, pool_size=3, pool_stride=1, pool_padding=1)
    x = conv_bn(input=x, num_filters=512, filter_size=3, padding=1, act='relu')
    x = conv_bn(input=x, num_filters=512, filter_size=3, padding=1, act='relu')
    x = conv_bn(input=x, num_filters=512, filter_size=3, padding=1)
    x = fluid.layers.dropout(x=x, dropout_prob=0.5)
    return x


def shallow_network(img):
    x = img
    x = conv_bn(input=x, num_filters=24, filter_size=5, padding=3, act='relu')
    x = fluid.layers.pool2d(input=x, pool_size=5, pool_type='avg', pool_stride=2)
    x = conv_bn(input=x, num_filters=24, filter_size=5, padding=3, act='relu')
    x = fluid.layers.pool2d(input=x, pool_size=5, pool_type='avg', pool_stride=2)
    x = conv_bn(input=x, num_filters=24, filter_size=5, padding=4, act='relu')
    x = fluid.layers.pool2d(input=x, pool_size=5, pool_type='avg', pool_stride=2)
    return x

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
```

通过上面实现的CrowdNet模型，它的结构如下图所示：

![CrowdNet模型结构](https://s1.ax1x.com/2020/06/05/tr4KSK.png)


# 训练模型
本项目使用的是百度公开的一个人流密度数据集，数据集下载链接：[https://aistudio.baidu.com/aistudio/datasetdetail/1917](https://aistudio.baidu.com/aistudio/datasetdetail/1917) ，下载之后，执行下面操作：
 - 把`train.json`文件存放在`data`目录
 - 把`test_new.zip`解压到`data`目录
 - 把`train_new.zip`解压到`data`目录
 
本项目提供了一个脚本`create_list.py`可以把百度公开的数据集数据标准文件生成本项目所需要的标注格式，通过执行脚本可以生成类似以下格式的数据列表，每一行的前面是图像路径，后面的是人的坐标点，中间用制表符`\t`分开。如果开发者要训练自己的数据集，将图像标注数据生成以下格式即可。
```
data/train/4c93da45f7dc854a31a4f75b1ee30056.jpg	[(171, 200), (365, 144), (306, 155), (451, 204), (436, 252), (600, 235)]
data/train/3a8c1ed636145f23e2c5eafce3863bb2.jpg	[(788, 205), (408, 250), (115, 233), (160, 261), (226, 225), (329, 161)]
data/train/075ed038030094f43f5e7b902d41d223.jpg	[(892, 646), (826, 763), (845, 75), (896, 260), (773, 752)]
```

模型的输入标签是一个密度图，那么如何通过标注数据生成一个密度图的，下面就来简单介绍一下。其实就是一些不同核的高斯滤波器生成的，得到的一个比输入图像小8倍的密度图。
```python
import json
import numpy as np
import scipy
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm as CM
import scipy
import scipy.spatial
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import os

# 图片预处理
def picture_opt(img, ann):
    # 缩放的图像大小
    train_img_size = (640, 480)
    gt = []
    size_x, size_y = img.size
    img = img.resize(train_img_size, Image.ANTIALIAS)

    for b_l in range(len(ann)):
        x = ann[b_l][0]
        y = ann[b_l][1]
        x = (x * train_img_size[0] / size_x) / 8
        y = (y * train_img_size[1] / size_y) / 8
        gt.append((x, y))

    img = np.array(img) / 255.0
    return img, gt

# 高斯滤波
def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1].ravel(), np.nonzero(gt)[0].ravel())))
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=2048)
    distances, locations = tree.query(pts, k=4)
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density


# 密度图处理
def ground(img, gt):
    imgs = img
    x = imgs.shape[0] / 8
    y = imgs.shape[1] / 8
    k = np.zeros((int(x), int(y)))
    for i in range(0, len(gt)):
        if int(gt[i][1]) < int(x) and int(gt[i][0]) < int(y):
            k[int(gt[i][1]), int(gt[i][0])] = 1
    img_sum = np.sum(k)
    k = gaussian_filter_density(k)
    return k, img_sum
```

读取一张图片，并经过缩放预处理，在这里图像没有经过装置，但是在训练过程中需要对图像执行装置`im.transpose()`操作，这样才符合PaddlePaddle的输入格式。
```python
# 读取数据列表
with open('data/data_list.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

line = lines[50]
img_path, gt = line.replace('\n', '').split('\t')
gt = eval(gt)
img = Image.open(img_path)
im, gt = picture_opt(img, gt)

print(im.shape)
plt.imshow(im)
```

![](https://s1.ax1x.com/2020/06/05/trn1Q1.png)

通过`ground()`函数将上面的图片生成一个密度图，密度图结果如下图所示。注意在输入PaddlePaddle的密度图是要经过装置的，因为图像的数据的输入是装置的，所以密度图也得装置。
```python
k, img_sum = ground(im, gt)
groundtruth = np.asarray(k)
groundtruth = groundtruth.astype('float32')

print("实际人数：", img_sum)
print("密度图人数：", np.sum(groundtruth))
print("密度图大小：", groundtruth.shape)

plt.imshow(groundtruth,cmap=CM.jet)
```

![人流密度图](https://s1.ax1x.com/2020/06/05/trnMW9.png)


## 训练程序
以下为`train.py`的代码，在训练中使用了平方差损失函数，其中损失值乘以`6e5`是为了不让输出的损失值太小。
```python
loss = fluid.layers.square_error_cost(input=map_out, label=label) * 6e5
loss = fluid.layers.mean(loss)
```

为了加快数据的读取，这里使用了异步数据读取方式，可以一边训练一边读取下一步batch的数据。
```python
py_reader = fluid.io.PyReader(feed_list=[images, label, img_num],
                              capacity=32,
                              iterable=True,
                              return_list=False)
py_reader.decorate_sample_list_generator(paddle.batch(reader.train_reader(data_list_file), batch_size=BATCH_SIZE),
                                         places=fluid.core.CPUPlace())
```

在训练前加上一个加载预训练模型的方法，如果之前的模型存在，就加载该模型，接着上一次的训练结果继续训练。
```python
if PERSISTABLES_MODEL_PATH is not None and os.path.exists(PERSISTABLES_MODEL_PATH):
    def if_exist(var):
        if os.path.exists(os.path.join(PERSISTABLES_MODEL_PATH, var.name)):
            print('loaded: %s' % var.name)
        return os.path.exists(os.path.join(PERSISTABLES_MODEL_PATH, var.name))


    fluid.io.load_vars(exe, PERSISTABLES_MODEL_PATH, main_program=fluid.default_main_program(), predicate=if_exist)
```

在执行训练前需要留意以下几个参数，需要根据自己的实际情况修改。当然如果开发者都是按照上面的操作，这里基本上不需要修改，但是`BATCH_SIZE`可能要修改一下，因为这个模型比较大，如何显存小的可能还有修改，以下是笔者在8G显存的环境下设置的。
```python
# 是否使用GPU
USE_CUDA = True
# 模型参数保存路径
PERSISTABLES_MODEL_PATH = 'persistables_model/'
# 预测模型保存路径
INFER_MODEL = 'infer_model/'
# 训练轮数
EPOCHS_SUM = 800
# Batch大小
BATCH_SIZE = 6
# 图像列表路径
data_list_file = 'data/data_list.txt'
```

最后执行`python train.py`开始训练模型。


# 预测
最通过执行`infer.py`可以把`data/test/`目录下的图像都进行预测，结果写入到`results.csv`文件中。

下面介绍预测的大概方式，通过加载训练过程中保存的预测模型，得到一个预测程序。
```python
import matplotlib.pyplot as plt
from matplotlib import cm as CM
import os
import numpy as np
import paddle.fluid as fluid
from PIL import Image

# 是否使用GPU
USE_CUDA = True
INFER_MODEL = 'infer_model/'

place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()
exe = fluid.Executor(place)

[inference_program,
 feed_target_names,
 fetch_targets] = fluid.io.load_inference_model(INFER_MODEL, exe)
```

读取一张待预测的图片。
```python
image_path = "data/test/00bdc7546131db72333c3e0ac9cf5478.jpg"
test_img = Image.open(image_path)
plt.imshow(test_img)
```

![](https://s1.ax1x.com/2020/06/05/trnQzR.png)

通过对图像进行预处理，输入到预测程序中，预测的结果有两个，第一个是密度图，第二个是估计人数，因为输出的估计是估计人数是一个带小数的值，所以要进行四舍五入。其实对密度图求和也是能够得到估计人数的。因为PaddlePaddle输出的密度图是经过转置的，所以在显示时需要再一次执行转置才能正常显示。
```python
test_img = test_img.resize((640, 480), Image.ANTIALIAS)
test_im = np.array(test_img) / 255.0
test_im = test_im.transpose().reshape(1, 3, 640, 480).astype('float32')

results = exe.run(program=inference_program,
                    feed={feed_target_names[0]: test_im},
                    fetch_list=fetch_targets)
density, quantity = results[0], results[1]
q = int(abs(quantity) + 0.5)

print("预测人数：", q)
plt.imshow(density[0][0].T,cmap=CM.jet)
```

![人流密度图](https://s1.ax1x.com/2020/06/05/trnKJJ.png)

# 模型下载

| 模型名称 | 所用数据集 | 下载地址 |
| :---: | :---: | :---: |
| 预训练模型 | 常规赛-人流密度预测数据集 | [点击下载](https://resource.doiduoyi.com/#24332i5) |
| 预测模型 | 常规赛-人流密度预测数据集 | [点击下载](https://resource.doiduoyi.com/#m761c1y) |


**创作不易，给个star吧**
