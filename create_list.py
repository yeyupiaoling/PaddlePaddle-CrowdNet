import json
import cv2
import numpy as np

# 原标注文件
annotation_file = 'data/train.json'
# 图像数据列表
data_list_file = 'data/data_list.txt'

# 把图片对应的标签装入字典
f = open(annotation_file, encoding='utf-8')
content = json.load(f)

f_list = open(data_list_file, 'w', encoding='utf-8')

# 图像路径替换
for j in range(len(content['annotations'])):
    content['annotations'][j]['name'] = content['annotations'][j]['name'].replace('stage1', 'data')


# 获取标注的xy
def get_xy(ann):
    gt = []
    for b_l in range(len(ann)):
        if 'w' in ann[b_l].keys():
            # 框转点
            x = (ann[b_l]['x'] + (ann[b_l]['x'] + ann[b_l]['w'])) / 2
            y = ann[b_l]['y'] + 20
            gt.append((x, y))
        else:
            x = ann[b_l]['x']
            y = ann[b_l]['y']
            gt.append((x, y))
    return gt


for ig_index in range(len(content['annotations'])):
    if len(content['annotations'][ig_index]['annotation']) == 2: continue
    if len(content['annotations'][ig_index]['annotation']) == 3: continue
    if len(content['annotations'][ig_index]['ignore_region']) == 2: continue
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
            img_path = content['annotations'][ig_index]['name']
            # 将忽略涂白色
            ig_cv_img = cv2.imread(img_path)
            pts = np.array(ig_list, np.int32)
            cv2.fillPoly(ig_cv_img, [pts], (0, 0, 0), cv2.LINE_AA)
            cv2.imwrite(img_path, ig_cv_img)
            annotation = content['annotations'][ig_index]['annotation']
            gt = get_xy(annotation)
            f_list.write('%s\t%s\n' % (img_path, gt))
    else:
        img_path = content['annotations'][ig_index]['name']
        annotation = content['annotations'][ig_index]['annotation']
        gt = get_xy(annotation)
        f_list.write('%s\t%s\n' % (img_path, gt))
