# import matplotlib
# matplotlib.use('tkagg')
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import os
##### set specific gpu #####
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from yolobodytest1 import YOLO_v3

class yolo_train:

    def __init__(self, annotation_file, anchor_boxes_file, classes, input_shape, batch_size):
        self.annotation_file = annotation_file
        self.anchor_boxes_file = anchor_boxes_file
        self.classes = classes
        self.num_classes = len(self.classes)
        if len(input_shape) == 2:
            self.input_shape = input_shape
        elif len(input_shape) == 3:
            self.input_shape = input_shape[:2]
        else: print("get right input shape.")
        ### other paramteres ###
        self.validation_rate = 0.1
        self.batch_size = batch_size

    def read_anchor(self):
        '''
        Read anchor_kmean results and reshape it into (9, 2)
        '''
        with open(self.anchor_boxes_file) as f:
            anchors_read = f.readline()
        anchors = np.array([float(x) for x in anchors_read.split(',')])
        anchors = anchors.reshape(-1, 2)
        return anchors

    def read_and_split_annotations(self):
        '''
        Read the annotation file line by line
        '''
        with open(self.annotation_file) as f:
            lines = f.readlines()
        np.random.seed(10)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_validation = int(len(lines) * self.validation_rate)
        num_training = len(lines) - num_validation
        lines_train = lines[:num_training]
        lines_validation = lines[num_training:]
        return lines_train, lines_validation

    def parse_annotation(self, lines_input, max_boxes = 20):
        '''
        read the annotation lines with number of each batch and
        transfer the lines into image and bounding box.

        Reshape the image into (416, 416, 3)
        Reshape the bounding box into (20, 5) 
        where 20 is the maximum boxes and 5 for x, y, w, h, class

        Therefore, the output of this function should be 
        image per batch(batch_size, 416, 416, 3)
        boxes per batch(batch_size, 20, 5)
        '''
        image_batch = []
        box_batch = []
        for i in range(self.batch_size):
            line = lines_input[i].split(' ')
            image_path = line[0]
            image = cv2.imread(image_path)
            h_original, w_original, channels = image.shape
            h_input, w_input = self.input_shape
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

            ##### resize the image to get std input #####
            new_image = np.ones(self.input_shape + (3,), dtype = np.float32) * 0.01
            scale = np.minimum(h_input/h_original, w_input/w_original)
            h_new = int(h_original * scale)
            w_new = int(w_original * scale)
            h_gap = (h_input - h_new) // 2
            w_gap = (w_input - w_new) // 2
            image_resize = cv2.resize(image, (w_new, h_new))
            new_image[h_gap: h_gap+h_new, w_gap: w_gap+w_new, :] = image_resize / 255.

            image_batch.append(new_image)

            ##### resize box to get corresponding boxes #####
            box_data = np.zeros((max_boxes,5))
            if len(box)>0:
                np.random.shuffle(box)
                if len(box)>max_boxes: box = box[:max_boxes]
                box[:, [0,2]] = box[:, [0,2]]*scale
                box[:, 0] += w_gap
                box[:, [1,3]] = box[:, [1,3]]*scale
                box[:, 1] += h_gap
                box_data[:len(box)] = box
            box_batch.append(box_data)

        image_batch = np.array(image_batch)
        box_batch = np.array(box_batch)

        return image_batch, box_batch


    def get_train_label(self, box_batch, anchors):
        '''
        parse the boxes into std label form in order to train the model.
        '''

        num_grid_format = 3 # they all call it "default setting"
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_grid_format==3 else [[3,4,5], [1,2,3]]

        inputshape = np.array(self.input_shape, dtype = np.int32)
        true_boxes = np.array(box_batch, dtype = np.float32)

        ### get the box's normalized center and width, height in order to calculate which grid its in
        box_center_xy = true_boxes[..., 0:2] + (true_boxes[..., 2:4] // 2)
        box_w_and_h = true_boxes[..., 2:4]
        # true_boxes[..., 0:2] = box_center_xy / inputshape
        # true_boxes[..., 2:4] = box_w_and_h / inputshape

        num_samples = true_boxes.shape[0]
        ### divide the grids into 3 groups according to the size of anchors ###
        grid_shapes = [inputshape//{0:32, 1:16, 2:8}[i] for i in range(num_grid_format)]
        y_true = [np.zeros((num_samples,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+self.num_classes), dtype=np.float32) for l in range(num_grid_format)]

        valid_boxes = box_w_and_h[..., 0] > 0
        num_anchors = len(anchors)

        for id_batch in range(num_samples):
            ### get all the valid boxes in the current batch
            ### recall that we limited the size of maximum boxes shown in pictures
            wh_boxes = box_w_and_h[id_batch, valid_boxes[id_batch]]
            if len(wh_boxes) == 0: continue
            max_anchor_index = np.zeros(len(wh_boxes))

            for id_box in range(len(wh_boxes)):
                w_ground_true, h_ground_true = wh_boxes[id_box]
                anchor_box_iou = np.zeros(num_anchors)
                for id_anchor in range(num_anchors):
                    w_anchor, h_anchor = anchors[id_anchor]
                    w_intersection = np.minimum(w_anchor, w_ground_true)
                    h_intersection = np.minimum(h_anchor, h_ground_true)
                    area_ground_true = w_ground_true * h_ground_true
                    area_anchor = w_anchor * h_anchor
                    area_intersection = w_intersection * h_intersection
                    iou = area_intersection / (area_ground_true + area_anchor - area_intersection)
                    anchor_box_iou[id_anchor] = iou
                anchor_best_fit = np.argmax(anchor_box_iou)
                '''
                Generating the true lables for training section.
                '''
                for an_format in range(num_grid_format):
                    if anchor_best_fit in anchor_mask[an_format]:
                        grid_one = np.floor((true_boxes[id_batch, id_box, 0]+true_boxes[id_batch, id_box, 2] // 2) / inputshape[0] * grid_shapes[an_format][0]).astype(np.int32)
                        grid_two = np.floor((true_boxes[id_batch, id_box, 1]+true_boxes[id_batch, id_box, 3] // 2) / inputshape[1] * grid_shapes[an_format][1]).astype(np.int32)
                        wh_anchor = anchor_mask[an_format].index(anchor_best_fit)
                        class_num = true_boxes[id_batch, id_box, 4].astype(np.int32)
                        y_true[an_format][id_batch, grid_one, grid_two, wh_anchor, 0:2] = box_center_xy[id_batch, id_box, ...] / inputshape
                        y_true[an_format][id_batch, grid_one, grid_two, wh_anchor, 2:4] = box_w_and_h[id_batch, id_box, ...] / inputshape
                        y_true[an_format][id_batch, grid_one, grid_two, wh_anchor, 4] = 1
                        y_true[an_format][id_batch, grid_one, grid_two, wh_anchor, 5 + class_num] = 1

        return y_true


    def train_data_generator(self):
        # plt.ion()
        # fig = plt.figure(figsize = (8,8))
        # ax1 = fig.add_subplot(111)
        anchors = self.read_anchor()
        lines_train, lines_validation = self.read_and_split_annotations()
        np.random.shuffle(lines_train)
        num_batch_train = len(lines_train) // self.batch_size

        for each_batch in range(10,num_batch_train):

            lines_train_batch = lines_train[self.batch_size*each_batch: self.batch_size*(each_batch+1)]

            img_b, box_b = self.parse_annotation(lines_train_batch)

            true_labels_b = self.get_train_label(box_b, anchors)

            # print("True Label SIZE L1",true_labels_b[0].shape)
            # print("True Label SIZE L2",true_labels_b[1].shape)
            # print("True Label SIZE L3",true_labels_b[2].shape)

            # for j in range(self.batch_size):
            #     img = img_b[j]
            #     box = box_b[j]
            #     plt.cla()
            #     ax1.clear()
            #     ax1.imshow(img)
            #     for idx in range(20):
            #         x = box[idx, 0]
            #         y = box[idx, 1]
            #         w = box[idx, 2]
            #         h = box[idx, 3]
            #         clas = box[idx, 4]
            #         if clas == 0:
            #             rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='blue',facecolor='none')
            #         elif clas == 1:
            #             rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='red',facecolor='none')
            #         ax1.add_patch(rect)
            #     # ax1.scatter(tx, ty)
            #     plt.show()
            #     plt.pause(1)

            yield img_b, true_labels_b


'''
ANOTHER IMPLEMENTATION.
'''

'''
PAY ATTENTION, THIS IS ONLY USED FOR DEBUGGING.
'''

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split(' ')
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    # if True:
    # resize image
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx = (w-nw)//2
    dy = (h-nh)//2
    image_data=0
    # if proc_img:
    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image_data = np.array(new_image) / 255.

    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        if len(box)>max_boxes: box = box[:max_boxes]
        box[:, [0,2]] = box[:, [0,2]]*scale
        box[:, 0] += dx
        box[:, [1,3]] = box[:, [1,3]]*scale 
        box[:, 1] += dy
        box_data[:len(box)] = box

    return image_data, box_data

def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true

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
        yield image_data, y_true

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

'''
THIS ONE 
'''
'''
OOOOOOOOO
'''

        
if __name__ == "__main__":
    annotation_file = '/home/azhang/Documents/experiments_in_VIVA/ylt/Yolov3_try_1906/coco_train_self_annotation.txt'
    anchor_boxes_file = '/home/azhang/Documents/experiments_in_VIVA/ylt/Yolov3_try_1906/anchor_kmean_9.txt'
    classes = [1, 3, 6, 8]
    input_shape = (416, 416)
    batch_size = 5
    num_epoches = 10

    yolo_t = yolo_train(annotation_file, anchor_boxes_file, classes, input_shape, batch_size)

    lines_train, lines_val = yolo_t.read_and_split_annotations()

    anchors = yolo_t.read_anchor()
    num_classes = len(classes)

    '''
    Test and try implementing my own yolo_v3
    '''
    learning_rate = 0.0001

    yolo = YOLO_v3(input_shape, batch_size, anchors, num_classes)
    X = yolo.build_input_holder()
    label = yolo.build_label_holder()

    pred = yolo.build_yolo_body(X)
    # pred2 = tf.reshape(pred[0], [-1, 13, 13, 3, 9])
    loss = yolo.loss_function(pred, label)

    loss_regularized = yolo.filters_regularization(loss, 5e-4)

    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        variables_unfrozen = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "output_first") + \
                            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "output_second") + \
                            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "output_third")
        training_op = optimizer.minimize(loss_regularized, var_list = variables_unfrozen)

    ##### INITIALIZATION #####
    init = tf.global_variables_initializer()

    ##### setting disposable memory of GPU #####
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        for epoch in range(num_epoches):
            for imgb, labelb in yolo_t.train_data_generator():
            # for imgb, labelb in data_generator_wrapper(lines_train, batch_size, input_shape, anchors, num_classes):
                ##### db #####
                # if temp1 is None:
                #     temp1 = imgb
                #     temp1 = temp1/255.
                #     temp2 = labelb
                # a = sess.run(loss_regularized, feed_dict = {X: imgb, 
                #                                 label[0]: labelb[0],
                #                                 label[1]: labelb[1],
                #                                 label[2]: labelb[2]})
                
                # print(a)

                sess.run(training_op, feed_dict = {X: imgb, 
                                                    label[0]: labelb[0],
                                                    label[1]: labelb[1],
                                                    label[2]: labelb[2]})
                a = loss.eval(feed_dict = {X: imgb, 
                                                label[0]: labelb[0],
                                                label[1]: labelb[1],
                                                label[2]: labelb[2]})

                print(a)
                # print([lossv[0], xy_lossv[0], wh_lossv[0], confidence_lossv[0], class_lossv[0]])
            # learning_rate *= 0.1


