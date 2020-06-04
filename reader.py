import json
import random
import cv2
from multiprocessing import cpu_count
import numpy as np
import paddle.fluid as fluid
import scipy
from PIL import Image, ImageEnhance
from scipy.ndimage.filters import gaussian_filter

# 把图片对应的标签装入字典
f = open('data/train.json', encoding='utf-8')
content = json.load(f)

# 图像路径替换
for j in range(len(content['annotations'])):
    content['annotations'][j]['name'] = content['annotations'][j]['name'].replace('stage1', 'data')


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
            if 'w' in ann[b_l].keys():
                # 框转点
                x = (ann[b_l]['x'] + (ann[b_l]['x'] + ann[b_l]['w'])) / 2
                y = ann[b_l]['y'] + 20
                x = (x * train_img_size[0] / size_x) / 8
                y = (y * train_img_size[1] / size_y) / 8
                gt.append((x, y))
            else:
                x = ann[b_l]['x']
                y = ann[b_l]['y']
                x = (x * train_img_size[0] / size_x) / 8
                y = (y * train_img_size[1] / size_y) / 8
                gt.append((x, y))
    elif r > 10:
        # 水平翻转
        size_x, size_y = img.size
        img = img.resize(train_img_size, Image.ANTIALIAS)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        for b_l in range(len(ann)):
            if 'w' in ann[b_l].keys():
                # 框转点
                x = (ann[b_l]['x'] + (ann[b_l]['x'] + ann[b_l]['w'])) / 2
                y = ann[b_l]['y'] + 20
                x = (train_img_size[0] - (x * train_img_size[0] / size_x)) / 8
                y = (y * train_img_size[1] / size_y) / 8
                gt.append((x, y))
            else:
                x = ann[b_l]['x']
                y = ann[b_l]['y']
                x = (train_img_size[0] - (x * train_img_size[0] / size_x)) / 8
                y = (y * train_img_size[1] / size_y) / 8
                gt.append((x, y))
    else:
        # 裁剪
        size_x, size_y = img.size
        img = img.crop((2, 2, size_x - 2, size_y - 2))
        img = img.resize(train_img_size, Image.ANTIALIAS)
        for b_l in range(len(ann)):
            if 'w' in ann[b_l].keys():
                x = (ann[b_l]['x'] + (ann[b_l]['x'] + ann[b_l]['w'])) / 2
                y = ann[b_l]['y'] + 20
                x = (x * train_img_size[0] / size_x) / 8
                y = (y * train_img_size[1] / size_y) / 8
                gt.append((x, y))
            else:
                x = ann[b_l]['x']
                y = ann[b_l]['y']
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
    path, ann = sample
    img = Image.open(path)
    im, gt = picture_opt(img, ann)
    k, img_sum = ground(im, gt)
    groundtruth = np.asarray(k)
    groundtruth = groundtruth.T.astype('float32')
    im = im.transpose().astype('float32')
    return im, groundtruth, img_sum


# 获取数据读取reader（忽略全部的忽略区）
def train_reader():
    def reader():
        random.shuffle(content['annotations'])
        for ig_index in range(len(content['annotations'])):
            # 忽略有忽略区的图片
            if len(content['annotations'][ig_index]['annotation']) == 2: continue
            if len(content['annotations'][ig_index]['annotation']) == 3: continue
            if content['annotations'][ig_index]['name'] == 'train/8538edb45aaf7df78336aa5b49001be6.jpg': continue
            if content['annotations'][ig_index]['name'] == 'train/377df0a7a9abc44e840e938521df3b54.jpg': continue
            if content['annotations'][ig_index]['ignore_region']: continue

            img_path = content['annotations'][ig_index]['name']
            ann = content['annotations'][ig_index]['annotation']
            yield img_path, ann

    return fluid.io.xmap_readers(train_mapper, reader, cpu_count(), 500)


# 获取数据读取reader
def train_reader2():
    def reader():
        random.shuffle(content['annotations'])
        for ig_index in range(len(content['annotations'])):
            if len(content['annotations'][ig_index]['annotation']) == 2: continue
            if len(content['annotations'][ig_index]['annotation']) == 3: continue
            if len(content['annotations'][ig_index]['ignore_region']) == 2:continue
            if content['annotations'][ig_index]['name'] == 'train/8538edb45aaf7df78336aa5b49001be6.jpg': continue
            if content['annotations'][ig_index]['name'] == 'train/377df0a7a9abc44e840e938521df3b54.jpg': continue
            # 判断是否存在忽略区
            if content['annotations'][ig_index]['ignore_region']:
                ig_list = []
                # 忽略区为一个
                if len(content['annotations'][ig_index]['ignore_region']) == 1:
                    ign_rge = content['annotations'][ig_index]['ignore_region'][0]
                    for ig_len in range(len(ign_rge)):
                        ig_list.append([ign_rge[ig_len]['x'], ign_rge[ig_len]['y']])
                    ig_cv_img = cv2.imread(content['annotations'][ig_index]['name'])
                    pts = np.array(ig_list, np.int32)
                    cv2.fillPoly(ig_cv_img, [pts], (0, 0, 0), cv2.LINE_AA)
                    ig_img = Image.fromarray(cv2.cvtColor(ig_cv_img, cv2.COLOR_BGR2RGB))
                    ann = content['annotations'][ig_index]['annotation']
                    ig_im, gt = picture_opt(ig_img, ann)
                    k, img_sum = ground(ig_im, gt)
                    groundtruth = np.asarray(k)
                    groundtruth = groundtruth.T.astype('float32')
                    ig_im = ig_im.transpose().astype('float32')
                    yield ig_im, groundtruth, img_sum
            else:
                img = Image.open(content['annotations'][ig_index]['name'])
                ann = content['annotations'][ig_index]['annotation']
                im, gt = picture_opt(img, ann)
                k, img_sum = ground(im, gt)
                groundtruth = np.asarray(k)
                groundtruth = groundtruth.T.astype('float32')
                im = im.transpose().astype('float32')
                yield im, groundtruth, img_sum

    return reader
