# !/usr/bin/env python
# -  *  - coding:utf-8 -  *  - 

'''
Project path & data path.
'''

# project path
path = '/home/wyundi/Project/DeepLearning/TensorFlow/ObjectDetection/YOLOv3_tensorflow2.0'

model_path = path + '/model_data'

train_path = model_path + '/train.txt'
classes_path = model_path + '/voc_classes.txt'
anchors_path = model_path + '/yolo_anchors.txt'

json_path = log_dir + '/model.json'

font_path = path + '/font/FiraMono-Medium.otf'

# VOC data path
data_path = '/home/wyundi/Project/DeepLearning/Data/VOC/VOC2012/VOCdevkit2012/VOC2012'

images_path = data_path + '/JPEGImages'
annotations_path = data_path + '/Annotations'
imageSets_path = data_path + '/ImageSets'
main_path = imageSets_path + '/Main'
imageSets_path_trainval = imageSets_path + '/Main/trainval.txt'
imageSets_path_test = imageSets_path + '/Main/test.txt'

# model path
weight_path = data_path + '/YOLOv3/weight/yolo_weight.h5'
log_dir = data_path + '/YOLOv3/logs/'