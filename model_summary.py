# !/usr/bin/env python
# -  *  - coding:utf-8 -  *  - 

import tensorflow as tf
from tensorflow import keras

from keras.models import load_model

path = '/home/wyundi/Project/DeepLearning/TensorFlow/ObjectDetection/YOLOv3'

model = load_model(path + '/model_data/yolo_weight.h5')

model.summary()