import numpy as np
import json
from collections import defaultdict

'''
Read the annotation from coco json file and transfer it
into the predefined file which provides convenience of
getting specific input files.
'''

annotationdictionary = defaultdict(list)
f = open('/home/adam/coco/annotation2017/annotations/instances_train2017.json', encoding='utf-8')
coco_data = json.load(f)

coco_annot = coco_data['annotations']
for each_ant in coco_annot:
    # Read each annotation and reorcognize them into images size.
    image_id = each_ant['image_id']
    image_plus_prefix = (12 - len(str(image_id)))*'0' + str(image_id)
    img_location = img = '/home/adam/coco/coco2017train/train2017/%s.jpg' % image_plus_prefix
    bbox = each_ant['bbox']
    class_belong = each_ant['category_id']

    '''
    Annotations chosen to be used for training:
        1 <- person
        2 <- bicycle
        3 <- car
        4 <- motorcycle
        5 <- airplane
        6 <- bus
        7 <- train
        8 <- truck
    
    For more information, go to https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    '''
    annotation_wanted = [1, 3, 6, 8]
    if class_belong in annotation_wanted:
        '''
        transfer:
            0 <- ppl
            1 <- car
            2 <- bus
            3 <- truck
        '''
        if class_belong == 1:
            class_transfer = 0
        elif class_belong == 3:
            class_transfer = 1
        elif class_belong == 6:
            class_transfer = 2
        else: class_transfer = 3
        annotationdictionary[img_location].append([bbox, class_transfer])

# Save it for using
f = open('coco_train_self_annotation.txt', 'w')
for img in annotationdictionary.keys():
    f.write(img)
    all_boxes = annotationdictionary[img]
    for each_box in all_boxes:
        x = int(each_box[0][0])
        y = int(each_box[0][1])
        w = int(each_box[0][2])
        h = int(each_box[0][3])

        box_write_format = " %d,%d,%d,%d,%d" % (x, y, w, h, each_box[1])
        f.write(box_write_format)
    f.write('\n')
f.close