import random
import scipy
import numpy as np
import paddle.fluid as fluid
from PIL import Image, ImageEnhance
from multiprocessing import cpu_count
from scipy.ndimage.filters import gaussian_filter


# 图片增强和预处理
def picture_opt(img, ann):
    # 缩放的图像大小
    train_img_size = (640, 480)
    # 随机图像处理
    prob = np.random.uniform(0, 1)
    if prob > 0.5:
        delta = np.random.uniform(-0.125, 0.125) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        delta = np.random.uniform(-0.5, 0.5) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        delta = np.random.uniform(-0.5, 0.5) + 1
        img = ImageEnhance.Color(img).enhance(delta)

    r = random.randint(0, 30)
    gt = []
    if r > 20:
        # 不做数据增强
        size_x, size_y = img.size
        img = img.resize(train_img_size, Image.ANTIALIAS)

        for b_l in range(len(ann)):
            x = ann[b_l][0]
            y = ann[b_l][1]
            x = (x * train_img_size[0] / size_x) / 8
            y = (y * train_img_size[1] / size_y) / 8
            gt.append((x, y))
    elif r > 10:
        # 水平翻转
        size_x, size_y = img.size
        img = img.resize(train_img_size, Image.ANTIALIAS)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        for b_l in range(len(ann)):
            x = ann[b_l][0]
            y = ann[b_l][1]
            x = (train_img_size[0] - (x * train_img_size[0] / size_x)) / 8
            y = (y * train_img_size[1] / size_y) / 8
            gt.append((x, y))
    else:
        # 裁剪
        size_x, size_y = img.size
        img = img.crop((2, 2, size_x - 2, size_y - 2))
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


# 图像预处理
def train_mapper(sample):
    path, gt = sample
    img = Image.open(path)
    im, gt = picture_opt(img, gt)
    k, img_sum = ground(im, gt)
    groundtruth = np.asarray(k)
    # 密度图转置符合PaddlePaddle输出
    groundtruth = groundtruth.T.astype('float32')
    # 图片转置符合PaddlePaddle输入
    im = im.transpose().astype('float32')
    return im, groundtruth, img_sum


# 获取数据读取reader
def train_reader(data_list_file):
    with open(data_list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    def reader():
        random.shuffle(lines)
        for line in lines:
            img_path, gt = line.replace('\n', '').split('\t')
            gt = eval(gt)
            yield img_path, gt

    return fluid.io.xmap_readers(train_mapper, reader, cpu_count(), 500)
