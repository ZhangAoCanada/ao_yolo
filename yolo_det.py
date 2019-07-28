import os
##### set specific gpu #####
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np
import cv2

from yolobodytest1 import YOLO_v3
from validation import yolo_validation


def resize_image(image, input_shape):
    '''
    Function: resize the image to the std yolo input.

    Args:
        image           ->      image read by opencv.
        input_shape     ->      std yolo input shape.

    Output:
        new_image       ->      resized image for detection.
    '''

    ##### read original image size #####
    h_original, w_original, channels = image.shape
    image_original_shape = (h_original, w_original)
    ##### read standard input size of yolo body #####
    h_input, w_input = input_shape

    ##### resize the image to the standard input size #####
    new_image = np.ones(input_shape + (3,), dtype = np.float32) * 0.01
    scale = np.minimum(h_input/h_original, w_input/w_original)
    h_new = int(h_original * scale)
    w_new = int(w_original * scale)
    h_gap = (h_input - h_new) // 2
    w_gap = (w_input - w_new) // 2
    image_resize = cv2.resize(image, (w_new, h_new))
    new_image[h_gap: h_gap+h_new, w_gap: w_gap+w_new, :] = image_resize / 255.

    return new_image, image_original_shape

def read_anchor(anchor_boxes_file):
    '''
    Function: read anchor_kmean results and reshape it into (9, 2)

    Args:
        anchor_boxes_file   ->      txt file which stores all the anchors used for training.

    return:
        anchors             ->      all anchors within a np.array()
    '''
    with open(anchor_boxes_file) as f:
        anchors_read = f.readline()
    anchors = np.array([float(x) for x in anchors_read.split(',')])
    anchors = anchors.reshape(-1, 2)

    return anchors

def transfer_prediction(pred, batch_size_train, num_classes):
    '''
    transfer prediction to size (batch_size, total length, 9)
    '''
    all_predictions = []
    for formid in range(3):
        current_pred = pred[formid]
        current_pred = np.reshape(current_pred, (batch_size_train, -1, 5 + num_classes)).astype(np.float32)
        # current_pred[..., :4] *= self.height
        all_predictions.append(current_pred)

    all_predictions = np.concatenate(all_predictions, axis = 1).astype(np.float32)
    # all_predictions[..., :4] *= self.height

    return all_predictions

def box_iou( b1, b2):
    '''
    Function: calculate the iou of two boxes.

    Args:
        b1 -> (1, 4) -> (xcen, ycen, w, h)
        b2 -> (batch_num, 4) -> (xcen, ycen, w, h)

    return:
        iou
    '''
    ##### transfer b1 to (x,y,x,y) ######
    b1_xymin = b1[:, 0:2] - b1[:, 2:4] / 2
    b1_xymax = b1[:, 0:2] + b1[:, 2:4] / 2

    ##### transfer b2 to (x,y,x,y) #####
    b2_xymin = b2[:, 0:2] - b2[:, 2:4] / 2
    b2_xymax = b2[:, 0:2] + b2[:, 2:4] / 2

    ##### intersection x,y #####
    inter_xymin = np.maximum(b1_xymin, b2_xymin).astype(np.float32)
    inter_xymax = np.minimum(b1_xymax, b2_xymax).astype(np.float32)
    inter_wh = np.maximum(inter_xymax - inter_xymin, 0.).astype(np.float32)

    inter_area = inter_wh[:, 0] * inter_wh[:, 1]

    ##### iou #####
    b1_wh = (b1_xymax - b1_xymin)
    b2_wh = (b2_xymax - b2_xymin)
    b1_area = b1_wh[:, 0] * b1_wh[:, 1]
    b2_area = b2_wh[:, 0] * b2_wh[:, 1]

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def NMS(pred_t, batch_size, objthreshold = 0.4, nmsthreshold = 0.6):
    '''
    Self Implementation of Non-maximum Suppression
    '''
    all_predictions = pred_t
    output = [None for _ in range(len(all_predictions))]
    for batid in range(batch_size):
        pred_current = all_predictions[batid]
        filter_obj = pred_current[:, 4] > objthreshold
        pred_current = pred_current[filter_obj]
        if len(pred_current) == 0:
            continue
        
        score = - pred_current[:, 4] * np.amax(pred_current[:, 5:], axis = -1)
        sort_index = np.argsort(score)
        pred_current_new = pred_current[sort_index]

        class_num = np.argmax(pred_current_new[:, 5:], axis = -1)
        class_num = np.reshape(class_num, (len(class_num), 1)).astype(np.float32)
        pred_new = np.concatenate((pred_current_new[:, :5], class_num), axis = -1).astype(np.float32)

        box_output = []
        while (len(pred_new) != 0):
            fisrt_box = pred_new[0, :4]
            fisrt_box = np.expand_dims(fisrt_box, axis = 0).astype(np.float32)
            iou_inside = box_iou(fisrt_box, pred_new[:, :4])

            invalid_overlap = iou_inside >= nmsthreshold
            same_class = pred_new[0, -1] == pred_new[:, -1]
            discarded_boxes = invalid_overlap & same_class
            
            discarded_objness = pred_new[discarded_boxes, 4:5]
            # if np.sum(discarded_objness) != 0:
            #     pred_new[0, :4] = np.sum(discarded_objness * pred_new[discarded_boxes, :4], axis = 0) / np.sum(discarded_objness)
            box_output.append(pred_new[0])
            pred_new = pred_new[~discarded_boxes]
        
        if len(box_output) != 0:
            box_output = np.stack(box_output)
            output[batid] = box_output

    return output

def boxes_true_size(NMS_output, orimg_shape, batch_size, input_shape):
    '''
    Function: transfer all the boxes to the original input image size.

    Args:
        NMS_output          ->          all the predictions after NMS filter
        orimg_shape         ->          the input image shape
        batch_size          ->          for image detection, the batch size is always 1
        input_shape         ->          input shape to the darknet 53

    Return:
        pred_boxes_sizeback ->          boxes sized back to original image
        pred_conf           ->          predicted objectness
        pred_class          ->          predicted class
    '''

    h_original, w_original = orimg_shape
    h_input, w_input = input_shape
    input_shape_ar = np.array(input_shape)
    scale = np.minimum(h_input/h_original, w_input/w_original)
    h_new = int(h_original * scale)
    w_new = int(w_original * scale)
    h_gap = (h_input - h_new) // 2
    w_gap = (w_input - w_new) // 2

    release_gap = np.array([w_gap, h_gap])

    for i_batch in range(batch_size):
        pred_bat = NMS_output[i_batch]
        pred_boxes_xycen = pred_bat[:, :2] * input_shape_ar
        pred_boxes_wh = pred_bat[:, 2:4] * input_shape_ar

        pred_xy = (pred_boxes_xycen - (pred_boxes_wh / 2) - release_gap) / scale
        pred_wh = pred_boxes_wh / scale

        pred_boxes_sizeback = np.concatenate([pred_xy, pred_wh], axis = -1)

        pred_conf = pred_bat[:, 4]
        pred_class = pred_bat[:, 5]

    return pred_boxes_sizeback, pred_conf, pred_class       

def main():
    '''
    Main function for running.
    
    Classes meanings:
        0 <- ppl
        1 <- car
        2 <- bus
        3 <- truck
    '''

    ##### get the image #####
    img_filename = "/home/azhang/Documents/experiments_in_VIVA/testimage/IMG_2006.jpg"
    image = cv2.imread(img_filename)

    ##### some basic parameters #####
    anchor_boxes_file = '/home/azhang/Documents/experiments_in_VIVA/ylt/Yolov3_try_1906/anchor_kmean_9.txt'
    classes = [0, 1, 2, 3]
    num_classes = len(classes)
    input_shape = (416, 416)
    batch_size_train = 1
    batch_size_val = 1

    ##### read the anchors #####
    anchors = read_anchor(anchor_boxes_file)

    ##### get the std input image #####
    n_img, orimg_shape = resize_image(image, input_shape)
    n_image = np.expand_dims(n_img, axis = 0)

    '''
    Re-build yolo body for detection.
    '''
    yolo = YOLO_v3(input_shape, batch_size_train, batch_size_val, anchors, num_classes)
    X = yolo.build_input_holder()
    label = yolo.build_label_holder()
    pred = yolo.build_yolo_body(X)
    pred_translated = yolo.translate_pred(pred)

    ##### results from pred #####
    # output_boxes, output_scores, output_classes = yolo.NMS_pred(pred, orimg_shape)

    ##### INITIALIZATION #####
    init = tf.global_variables_initializer()

    ##### setting disposable memory of GPU #####
    gpu_options = tf.GPUOptions(allow_growth=True)

    ##### save the model ######
    saver = tf.train.Saver()
    saver_filename = "/home/azhang/Documents/experiments_in_VIVA/Yolov3_try_1906/model/yolo_test_first_third.ckpt"

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver.restore(sess, saver_filename)
        # sess.run(init)
        
        pred_t = sess.run(pred_translated, feed_dict = {X: n_image})

        pred_ft = transfer_prediction(pred_t, batch_size_train, num_classes)

        outputpred = NMS(pred_ft,  batch_size_train)

        boxes, confs, classes = boxes_true_size(outputpred, orimg_shape, batch_size_train, input_shape)

    ##### draw picture and boxes #####
    # plt.ion()
    fig = plt.figure(figsize = (8,8))
    ax1 = fig.add_subplot(111)
    if len(boxes) == 0:
        print("No object detected, please check your model.")
    else:
        hm_boxes = len(boxes)
        ax1.imshow(image[..., ::-1])
        for idx in range(hm_boxes):
            x = int(boxes[idx, 0])
            y = int(boxes[idx, 1])
            w = int(boxes[idx, 2])
            h = int(boxes[idx, 3])
            clas = classes[idx]
            sco = confs[idx]
            if clas == 0:
                rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='blue',facecolor='none')
            elif clas == 1:
                rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='red',facecolor='none')
            elif clas == 2:
                rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='yellow',facecolor='none')
            elif clas == 3:
                rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='black',facecolor='none')
            ax1.add_patch(rect)
        # ax1.scatter(tx, ty)
        plt.show()
    

if __name__ == "__main__":
    main()
