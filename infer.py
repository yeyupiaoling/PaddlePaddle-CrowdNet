import csv
import os
import zipfile

import cv2
import numpy as np
import paddle.fluid as fluid
from PIL import Image

test_zfile = zipfile.ZipFile("data/test_new.zip")
l_test = []
for test_fname in test_zfile.namelist()[1:]:
    l_test.append(test_fname)

data_dict = {}
use = True
place = fluid.CUDAPlace(0) if use else fluid.CPUPlace()
exe = fluid.Executor(place)

[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model('save_model/', exe)

for index in range(len(l_test)):
    ig_cv_img = Image.open(os.path.join('data', l_test[index]))
    test_img = ig_cv_img.resize((640, 480), Image.ANTIALIAS)
    test_im = np.array(test_img)
    test_im = test_im / 255.0
    test_im = test_im.transpose().reshape(1, 3, 640, 480).astype('float32')
    l_test[index] = l_test[index].lstrip('test').lstrip('/')

    results = exe.run(inference_program,
                      feed={feed_target_names[0]: test_im},
                      fetch_list=fetch_targets)

    people = np.sum(results)
    print(index, l_test[index], int(people))
    data_dict[l_test[index]] = int(people)

with open('results.csv', 'w') as csvfile:
    fieldnames = ['id', 'predicted']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for k, v in data_dict.items():
        writer.writerow({'id': k, 'predicted': v})
