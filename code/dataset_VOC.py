# !/usr/bin/env python
# -  *  - coding:utf-8 -  *  - 
# https://zhuanlan.zhihu.com/p/31468683

def get_dataset():

    '''

    dataset element:
        filepath (xml file)
        img_filename
        width
        height
        bboxes:
            {class, x1, x2, y1, y2, difficult}
            ...
        imageset: trainval / test / trainval

    '''

    import os
    import xml.etree.ElementTree as ET
    from cv2 import cv2

    from path import data_path, images_path, annotations_path, imageSets_path, \
                     main_path, imageSets_path_trainval, imageSets_path_test

    # 变量存储信息
    all_imgs = []

    classes_count = {}
    class_mapping = {}

    visualise = False
    # visualise = True

    print('Parsing annotation files')    

    # 获取训练集和测试集
    trainval_files = []
    test_files = []

    try:
        with open(imageSets_path_trainval) as f:
            for line in f:
                trainval_files.append(line.strip() + '.jpg')
    except Exception as e:
        print(e)

    try:
        with open(imageSets_path_test) as f:
            for line in f:
                test_files.append(line.strip() + '.jpg')
    except Exception as e:
        if data_path[-7:] == 'VOC2012':
            # print('VOC2012 has no test')
            pass
        else:
            print(e)

    # 获取数据集标注信息
    annots = [annotations_path + '/' + s for s in os.listdir(annotations_path)]

    for annot in annots:
        try:
            et = ET.parse(annot)
            element = et.getroot()

            element_objs = element.findall('object')
            element_name = element.find('filename').text
            element_filename = element.find('filename').text
            element_width = int(element.find('size').find('width').text)
            element_height = int(element.find('size').find('height').text)

            if len(element_objs) > 0:
                annot_data = {'filepath': annot,
                                'img_filename': element_name,
                                'width': element_width,
                                'height': element_height,
                                'bboxes': []}
                if element_filename in trainval_files:
                    annot_data['imageset'] = 'trainval'
                elif element_filename in test_files:
                    annot_data['imageset'] = 'test'
                else:
                    annot_data['imageset'] = 'trainval'

            for obj in element_objs:
                class_name = obj.find('name').text
                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1

                if class_name not in class_mapping:
                    class_mapping[class_name] = len(class_mapping)

                obj_bbox = obj.find('bndbox')
                x1 = int(round(float(obj_bbox.find('xmin').text)))
                y1 = int(round(float(obj_bbox.find('ymin').text)))
                x2 = int(round(float(obj_bbox.find('xmax').text)))
                y2 = int(round(float(obj_bbox.find('ymax').text)))
                difficulty = int(obj.find('difficult').text) == 1
                
                annot_data['bboxes'].append({'class':class_name,'x1':x1,'x2':x2,'y1':y1,'y2':y2,'difficult':difficulty})
            
            all_imgs.append(annot_data)
            
            if visualise:
                print(annot_data['img_filename'])
                img = cv2.imread(images_path + '/' + annot_data['img_filename'])

                for bbox in annot_data['bboxes']:
                    print(bbox['class'])
                    cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 0, 255))

                cv2.imshow('img', img)
                cv2.waitKey(0)

        except Exception as e:
            print(e)
            continue
    
    return all_imgs

def _main():
    dataset = get_dataset()
    for i in range(3):
        print(dataset[i]['img_filename'])

if __name__ == '__main__':
    _main()