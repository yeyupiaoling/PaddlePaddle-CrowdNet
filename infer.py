import csv
import os
import numpy as np
import paddle.fluid as fluid
from PIL import Image
from tqdm import tqdm

# 是否使用GPU
USE_CUDA = True
INFER_MODEL = 'infer_model/'

place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()
exe = fluid.Executor(place)

[inference_program,
 feed_target_names,
 fetch_targets] = fluid.io.load_inference_model(INFER_MODEL, exe)


def infer(path):
    ig_cv_img = Image.open(path)
    test_img = ig_cv_img.resize((640, 480), Image.ANTIALIAS)
    test_im = np.array(test_img) / 255.0
    test_im = test_im.transpose().reshape(1, 3, 640, 480).astype('float32')

    results = exe.run(program=inference_program,
                      feed={feed_target_names[0]: test_im},
                      fetch_list=fetch_targets)
    density, quantity = results[0], results[1]
    q = int(abs(quantity) + 0.5)
    return q


if __name__ == '__main__':
    # 预测结果都放在这个字典中
    data_dict = {}
    test_path = 'data/test/'
    images = os.listdir(test_path)
    for image in tqdm(images):
        image_path = os.path.join(test_path, image)
        q = infer(image_path)
        data_dict[image[:-4]] = int(q)

    with open('results.csv', 'w', newline='') as csvfile:
        fieldnames = ['id', 'predicted']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for k, v in data_dict.items():
            writer.writerow({'id': k, 'predicted': v})
