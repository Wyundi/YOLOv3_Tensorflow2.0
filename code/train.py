# !/usr/bin/env python
# -  *  - coding:utf-8 -  *  - 

import numpy as np

import tensorflow as tf
from tensorflow import keras

import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Lambda
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# 路径
from path import path
from path import train_path, weight_path, classes_path, anchors_path
from path import log_dir, json_path

# 功能包
from utils import get_classes, get_anchors
from utils import get_random_data

# Model
from model import yolo_body, yolo_loss
from model import preprocess_true_boxes

def _main():
    # 显存分配 
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # 模型参数
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)

    # 创建模型
    model = create_model(num_classes, weight_path, anchors, input_shape)

    # Callback
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + '/' + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # 打乱/分割训练集
    val_split = 0.1
    with open(train_path) as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if not True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 64
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=100,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        
        model.save_weights(log_dir + '/trained_weights_stage_1.h5')
        print("Save path: ", log_dir , '/trained_weights_stage_1.h5')
    else:
        new_weight = log_dir + '/ep071-loss19.917-val_loss10.860.h5'
        assert new_weight.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        model.load_weights(new_weight)
        

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.

    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')
        
        batch_size = 4 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=200,
            initial_epoch=100,
            callbacks=[logging, checkpoint])
            # callbacks=[logging, checkpoint, reduce_lr, early_stopping])

        model.save_weights(log_dir + '/trained_weights_final.h5')
        print("Save path: ", log_dir, '/trained_weights_final.h5')

    # Further training if needed.

def create_model(num_classes, weights_path, anchors, input_shape, load_pretrained=True, freeze_body=2):
    '''create the training model'''
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    # 真值的形状
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    # 冻结层
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    # 损失函数
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
                [*model_body.output, *y_true])

    # 模型
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''check if there are any wrong number'''
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()