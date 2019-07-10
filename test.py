import json
from PIL import Image
import numpy as np
import random

f = open('data/train.json', encoding='utf-8')
content = json.load(f)

for j in range(len(content['annotations'])):
    content['annotations'][j]['name'] = content['annotations'][j]['name'].replace('stage1', 'data')

img_path = content['annotations'][4]['name']
ann = content['annotations'][4]['annotation']

img = Image.open(img_path)

r = random.randint(0, 30)

if r > 20:
    size_x, size_y = img.size
    train_img_size = (640, 480)
    img = img.resize(train_img_size, Image.ANTIALIAS)

    for b_l in range(len(ann)):
        if 'w' in ann[b_l].keys():
            x = (ann[b_l]['x'] + (ann[b_l]['x'] + ann[b_l]['w'])) / 2
            y = ann[b_l]['y'] + 20
            x = (x * 640 / size_x)
            y = (y * 480 / size_y)
        else:
            x = ann[b_l]['x']
            y = ann[b_l]['y']
            x = (x * 640 / size_x)
            y = (y * 480 / size_y)
        for i in range(100):
            img.putpixel((int(x + random.randint(0, 5)), int(y + random.randint(0, 5))), (0, 255, 0))
elif r > 10:
    size_x, size_y = img.size
    train_img_size = (640, 480)
    img = img.resize(train_img_size, Image.ANTIALIAS)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    for b_l in range(len(ann)):
        if 'w' in ann[b_l].keys():
            x = (ann[b_l]['x'] + (ann[b_l]['x'] + ann[b_l]['w'])) / 2
            y = ann[b_l]['y'] + 20
            x = 640 - (x * 640 / size_x)
            y = (y * 480 / size_y)
        else:
            x = ann[b_l]['x']
            y = ann[b_l]['y']
            x = 640 - (x * 640 / size_x)
            y = (y * 480 / size_y)
        for i in range(100):
            img.putpixel((int(x + random.randint(0, 5)), int(y + random.randint(0, 5))), (0, 255, 0))
    img.save('test.png')
else:
    size_x, size_y = img.size
    print(size_x, size_y)
    img = img.crop((2, 2, size_x - 2, size_y - 2))
    train_img_size = (640, 480)
    img = img.resize(train_img_size, Image.ANTIALIAS)
    for b_l in range(len(ann)):
        if 'w' in ann[b_l].keys():
            x = (ann[b_l]['x'] + (ann[b_l]['x'] + ann[b_l]['w'])) / 2
            y = ann[b_l]['y'] + 20
            x = (x * 640 / size_x)
            y = (y * 480 / size_y)
        else:
            x = ann[b_l]['x']
            y = ann[b_l]['y']
            x = (x * 640 / size_x)
            y = (y * 480 / size_y)
        for i in range(100):
            img.putpixel((int(x + random.randint(0, 5)), int(y + random.randint(0, 5))), (0, 255, 0))

img.save('test.png')


# 图片操作
def picture_opt(img, ann):
    r = random.randint(11, 30)

    gt = []
    if r > 20:
        size_x, size_y = img.size
        train_img_size = (640, 480)
        img = img.resize(train_img_size, Image.ANTIALIAS)

        for b_l in range(len(ann)):
            if 'w' in ann[b_l].keys():
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
        size_x, size_y = img.size
        train_img_size = (640, 480)
        img = img.resize(train_img_size, Image.ANTIALIAS)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        for b_l in range(len(ann)):
            if 'w' in ann[b_l].keys():
                x = (ann[b_l]['x'] + (ann[b_l]['x'] + ann[b_l]['w'])) / 2
                y = ann[b_l]['y'] + 20
                x = 640 - (x * 640 / size_x) / 8
                y = (y * 480 / size_y) / 8
                gt.append((x, y))
            else:
                x = ann[b_l]['x']
                y = ann[b_l]['y']
                x = 640 - (x * 640 / size_x) / 8
                y = (y * 480 / size_y) / 8
                gt.append((x, y))
    else:
        size_x, size_y = img.size
        img = img.crop((2, 2, size_x - 2, size_y - 2))
        train_img_size = (640, 480)
        img = img.resize(train_img_size, Image.ANTIALIAS)
        for b_l in range(len(ann)):
            if 'w' in ann[b_l].keys():
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

    img = np.array(img)
    img = img / 255.0
    return img, gt