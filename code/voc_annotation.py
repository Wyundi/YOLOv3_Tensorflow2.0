# !/usr/bin/env python
# -  *  - coding:utf-8 -  *  - 

import os

import dataset_VOC as init
from utils import get_classes
from path import classes_path, images_path, train_path

# 获取数据集
dataset = init.get_dataset()

class_names = get_classes(classes_path)
num_classes = len(class_names)

classes_dir = {}
for i in range(num_classes):
    classes_dir[class_names[i]] = i

# xml文件转txt
if os.path.exists(train_path):
    os.remove(train_path)

os.mknod(train_path) 

f = open(train_path, 'a')

for i in range(len(dataset)):
# for i in range(10):
    name = dataset[i]['img_filename']
    box = dataset[i]['bboxes']
    length = len(box)

    line = images_path + '/' + name
    
    for j in range(length):
        classes = classes_dir[box[j]['class']]
        x1, y1, x2, y2 = box[j]['x1'], box[j]['y1'], box[j]['x2'], box[j]['y2']
        line += ' {},{},{},{},{}' .format(x1, y1, x2, y2, classes)

    line += '\n'
    f.write(line)