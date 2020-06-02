import json
import random
from multiprocessing import cpu_count
import numpy as np
import paddle
import scipy
from PIL import Image, ImageEnhance
from scipy.ndimage.filters import gaussian_filter

# 把图片对应的标签装入字典
f = open('data/train.json', encoding='utf-8')
content = json.load(f)

# 图像路径替换
for j in range(len(content['annotations'])):
    content['annotations'][j]['name'] = content['annotations'][j]['name'].replace('stage1', 'data')


# 图片增强和御处理
def picture_opt(img, ann):
    r = random.randint(0, 30)

    gt = []
    if r > 20:
        # 不做数据增强
        size_x, size_y = img.size
        train_img_size = (640, 480)
        img = img.resize(train_img_size, Image.ANTIALIAS)

        for b_l in range(len(ann)):
            if 'w' in ann[b_l].keys():
                # 框转点
                x = (ann[b_l]['x'] + (ann[b_l]['x'] + ann[b_l]['w'])) / 2
                y = ann[b_l]['y'] + 20
                x = (x * 640 / size_x) / 8
                y = (y * 480 / size_y) / 8
                gt.append((x, y))
            else:
                x = ann[b_l]['x']
                y = ann[b_l]['y']
                x = (x * 640 / size_x) / 8
                y = (y * 480 / size_y) / 8
                gt.append((x, y))
    elif r > 10:
        # 水平翻转
        size_x, size_y = img.size
        train_img_size = (640, 480)
        img = img.resize(train_img_size, Image.ANTIALIAS)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        for b_l in range(len(ann)):
            if 'w' in ann[b_l].keys():
                # 框转点
                x = (ann[b_l]['x'] + (ann[b_l]['x'] + ann[b_l]['w'])) / 2
                y = ann[b_l]['y'] + 20
                x = (640 - (x * 640 / size_x)) / 8
                y = (y * 480 / size_y) / 8
                gt.append((x, y))
            else:
                x = ann[b_l]['x']
                y = ann[b_l]['y']
                x = (640 - (x * 640 / size_x)) / 8
                y = (y * 480 / size_y) / 8
                gt.append((x, y))
    else:
        # 随机裁剪
        size_x1, size_y1 = img.size
        r_size_x1 = random.randint(0, 30)
        r_size_y1 = random.randint(0, 30)
        r_size_x2 = random.randint(0, 30)
        r_size_y2 = random.randint(0, 30)
        img = img.crop((r_size_x1, r_size_y1, size_x1 - r_size_x2, size_y1 - r_size_y2))
        size_x, size_y = img.size
        train_img_size = (640, 480)
        img = img.resize(train_img_size, Image.ANTIALIAS)
        for b_l in range(len(ann)):
            if 'w' in ann[b_l].keys():
                # 框转点
                x = (ann[b_l]['x'] + (ann[b_l]['x'] + ann[b_l]['w'])) / 2
                y = ann[b_l]['y'] + 20
                if r_size_x1 < x < (size_x1 - r_size_x2) and r_size_y1 < y < (size_y1 - r_size_y2):
                    x = x - r_size_x1
                    y = y - r_size_y1
                    x = (x * 640 / size_x) / 8
                    y = (y * 480 / size_y) / 8
                    gt.append((x, y))
            else:
                x = ann[b_l]['x']
                y = ann[b_l]['y']
                if r_size_x1 < x < (size_x1 - r_size_x2) and r_size_y1 < y < (size_y1 - r_size_y2):
                    x = x - r_size_x1
                    y = y - r_size_y1
                    x = (x * 640 / size_x) / 8
                    y = (y * 480 / size_y) / 8
                    gt.append((x, y))

    img = np.array(img)
    img = img / 255.0
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
            sigma = 10
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

    im, gt = picture_opt(img, ann)
    k, img_sum = ground(im, gt)
    groundtruth = np.asarray(k)
    groundtruth = groundtruth.T.astype('float32')
    im = im.transpose().astype('float32')
    return im, groundtruth, img_sum


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

    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), 500)
