# !/usr/bin/env python
# -  *  - coding:utf-8 -  *  - 

import numpy as np

import tensorflow as tf
from tensorflow import keras

from keras import Model
from keras.layers import Input
from keras import backend as K
from keras.models import load_model

from cv2 import cv2

from model import yolo_body, yolo_eval
from train import create_model

from utils import get_anchors, get_classes
from utils import letterbox_image, letterbox_frame

from path import path
from path import weight_path, classes_path, anchors_path, log_dir, json_path
from path import images_path
from path import font_path

from PIL import Image, ImageFont, ImageDraw
import colorsys

import os
import multiprocessing as mp
import time

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
    score = 0.3
    iou = 0.45

    # 获取分类框颜色
    colors = get_colors(num_classes)

    # 启动子进程
    frame_queue = mp.Queue()
    result_queue = mp.Queue()
    label_queue = mp.Queue()
    video_redad = mp.Process(target = opencv_process, args = (frame_queue, result_queue, label_queue))

    video_redad.start()

    while(True):
        if frame_queue.qsize() >= 1:
            frame = frame_queue.get()
        else:
            print('no video')
            frame = np.zeros([416, 416, 3])
            
        # 检测
        image_data, raw_image_shape = process_frame(frame, input_shape)
        res = detect(image_data, raw_image_shape, model, model_param, score, iou)

        if res != []:
            result_queue.put(res)

        if not label_queue.empty():
            if label_queue.get() == None:
                break

    while not label_queue.empty():
        label_queue.get()


    video_redad.join()

    print('Main process done')


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

    train_weights_path = log_dir + '/trained_weights_final.h5'

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

def process_frame(frame, input_shape):
    """
    将输入的帧处理成模型的标准输入形状
    """
    image_width = frame.shape[0]
    image_hight = frame.shape[1]
    image_shape = (image_hight, image_width)

    if input_shape != (None, None):
        assert input_shape[0]%32 == 0, 'Multiples of 32 required'
        assert input_shape[1]%32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_frame(frame, tuple(reversed(input_shape)))
    else:
        new_image_size = (image_width - (image_width % 32),
                            image_hight - (image_hight % 32))
        boxed_image = letterbox_frame(frame, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')

    # 归一化
    image_data /= 255.
    # Add batch dimension. (w, h, 3) -> (m, w, h, 3)
    image_data = np.expand_dims(image_data, 0)

    # print(image_data.shape)

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
    # print(grid1.shape, grid2.shape, grid3.shape)
    # print(grid1[0, 7, 5, :25])

    # predict

    boxes, scores, classes = yolo_eval(data, anchors,
                num_classes, image_shape,
                score_threshold=score_threshold, iou_threshold=iou_threshold)

    result = [boxes, scores, classes]
    return result

def output(image_shape, boxes, scores, classes, class_names):
    """
    processing output data
    """

    res = []

    for i, c in reversed(list(enumerate(classes))):
        predicted_class = class_names[c]
        box = boxes[i]
        score = scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        '''
        识别框调整：
            纵向：数值增大，识别框向下移动
            横向：数值增大，识别框向右移动
        url：-300， 500
        视频：-200， 250
        紧急救援：-220， 200
        '''

        height_dv = -220
        width_dv = 200

        # height_dv = -300
        # width_dv = 500
    
        top, left, bottom, right = box
        top = max(0, np.floor(top + height_dv).astype('int32'))
        left = max(0, np.floor(left + width_dv).astype('int32'))
        bottom = min(image_shape[1], np.floor(bottom + height_dv).astype('int32'))
        right = min(image_shape[0], np.floor(right + width_dv).astype('int32'))

        res_single = [label, left, top, right, bottom]
        res.append(res_single)

    return res

def detect(image_data, raw_image_shape, model, model_param, score, iou):
    class_names, num_classes, anchors, input_shape = model_param

    # 预测
    data = model.predict(image_data)
    boxes, scores, classes = predict(data, raw_image_shape, model, model_param, score, iou)
    # print('Found {} boxes form {}'.format(len(boxes), 'img'))

    # 输出
    res = output(raw_image_shape, boxes, scores, classes, class_names)

    if res != []:
        print(res)

    return res

def opencv_process(frame_queue, result_queue, label_queue):
    # 视频流
    url = "rtsp://admin:tiyoa000@192.168.0.64/Streaming/Channels/1"
    cap = cv2.VideoCapture(url)

    cap = cv2.VideoCapture('/home/wyundi/Videos/190319222227698228.mp4')

    while(True):
        ret, frame = cap.read()

        # 调整帧率
        time.sleep(0.03)

        if frame_queue.qsize() < 2:
            frame_queue.put(frame)

        if not result_queue.empty():
            res = result_queue.get()

            for i in range(len(res)):
                label, left, top, right, bottom = res[i]
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 3)

        cv2.namedWindow("capture", 0)
        # cv2.resizeWindow("capture", 1080, 640)    # 设置长和宽

        cv2.imshow('capture', frame)

        if cv2.waitKey(1) == ord('q'):
            label_queue.put(None)
            break

    cap.release()
    cv2.destroyAllWindows()

    while not frame_queue.empty():
        frame_queue.get()

    print('Sub process done')


if __name__ == '__main__':
    _main()