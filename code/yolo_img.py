# !/usr/bin/env python
# -  *  - coding:utf-8 -  *  - 

import numpy as np

import tensorflow as tf
from tensorflow import keras

from keras import Model
from keras.layers import Input
from keras import backend as K
from keras.models import load_model

from model import yolo_body, yolo_eval
from train import create_model

from utils import get_anchors, get_classes, letterbox_image

from path import path
from path import weight_path, classes_path, anchors_path, log_dir, json_path
from path import images_path
from path import font_path

from PIL import Image, ImageFont, ImageDraw
import colorsys

import os

def _main():
    # 显存分配 
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # 屏蔽warning
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # 加载模型
    model, model_param = load()
    class_names, num_classes, anchors, input_shape = model_param
    # model.summary()

    # 预测参数调整
    score = 0.7
    iou = 0.5

    # 获取分类框颜色
    colors = get_colors(num_classes)

    # 读取图像 + 预处理, image -> (1, 416, 416, 3)
    img_path = '/home/wyundi/Pictures/cat.jpg'
    image_data, image_shape = process_image(img_path, input_shape)

    # 预测
    data = model.predict(image_data)
    boxes, scores, classes = predict(data, image_shape, model, model_param, score, iou)
    print('Found {} boxes form {}'.format(len(boxes), 'img'))

    # 输出
    image = Image.open(img_path)
    output(image, boxes, scores, classes, class_names)


def load():
    """
    加载模型，回传模型和模型参数
    """

    # 模型参数
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    anchors = get_anchors(anchors_path)
    num_anchors = len(anchors)

    input_shape = (416, 416)

    train_weights_path = log_dir + '/ep169-loss17.356-val_loss6.844.h5'

    # 获取模型结构，加载权重
    image_input = Input(shape=(None, None, 3))

    model = yolo_body(image_input, num_anchors//3, num_classes)
    print('Get YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    model.load_weights(train_weights_path, by_name=True, skip_mismatch=True)
    print('Load weights {}.'.format(train_weights_path))

    # 回传参数
    param = [class_names, num_classes, anchors, input_shape]

    return model, param

def process_image(img_path, input_shape):
    """
    将输入的图片处理成模型的标准输入形状
    """

    image = Image.open(img_path)
    image_shape = (image.width, image.height)

    # 改变image的形状
    if input_shape != (None, None):
        assert input_shape[0]%32 == 0, 'Multiples of 32 required'
        assert input_shape[1]%32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(input_shape)))
    else:
        new_image_size = (image.width - (image.width % 32),
                            image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')

    # 归一化
    image_data /= 255.
    # Add batch dimension. (w, h, 3) -> (m, w, h, 3)
    image_data = np.expand_dims(image_data, 0)

    print(image_data.shape)

    return image_data, image_shape

def get_colors(num_classes):
    # Generate colors for drawing bounding boxes.
    # 生成随机颜色列表，以对应不同的框
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]

    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255),
                    int(x[1] * 255), int(x[2] * 255)),colors))

    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default

    return colors

def predict(data, image_shape, 
            model, model_param,
            score_threshold = .6,
            iou_threshold = .5,
            max_boxes = 20):
    """
    yolo_eval:
    Evaluate YOLO model on given input and return filtered boxes.
    """

    class_names, num_classes, anchors, input_shape = model_param

    # data shape = [(1, 13, 13, 75), (1, 26, 26, 75), (1, 52, 52, 75)]
    grid1, grid2, grid3 = np.array(data[0]), np.array(data[1]), np.array(data[2])
    print(grid1.shape, grid2.shape, grid3.shape)
    # print(grid1[0, 7, 5, :25])

    # predict

    boxes, scores, classes = yolo_eval(data, anchors,
                num_classes, image_shape,
                score_threshold=score_threshold, iou_threshold=iou_threshold)

    result = [boxes, scores, classes]
    return result

def output(image, boxes, scores, classes, class_names):
    """
    processing output data
    """

    for i, c in reversed(list(enumerate(classes))):
        predicted_class = class_names[c]
        box = boxes[i]
        score = scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))


if __name__ == '__main__':
    _main()