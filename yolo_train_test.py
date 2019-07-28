# import matplotlib
# matplotlib.use('tkagg')
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import os
##### set specific gpu #####
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from yolobodytest1 import YOLO_v3
from validation import yolo_validation
# from keras_yolo_body import yolo_body, tiny_yolo_body, yolo_loss, yolo_eval

class yolo_train:

    def __init__(self, annotation_file, anchor_boxes_file, classes, input_shape):
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
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_validation = int(len(lines) * self.validation_rate)
        num_training = len(lines) - num_validation
        lines_train = lines[:num_training]
        lines_validation = lines[num_training:]
        return lines_train, lines_validation

    def parse_annotation(self, lines_input, batch_size, max_boxes = 20):
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
        original_img = []
        image_batch = []
        box_batch = []
        for i in range(batch_size):
            line = lines_input[i].split(' ')
            image_path = line[0]
            image = cv2.imread(image_path)
            original_img.append(image_path)
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
        original_img = np.array(original_img)

        return image_batch, box_batch, original_img


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
                        y_true[an_format][id_batch, grid_two, grid_one, wh_anchor, 0:2] = box_center_xy[id_batch, id_box, ...] / inputshape
                        y_true[an_format][id_batch, grid_two, grid_one, wh_anchor, 2:4] = box_w_and_h[id_batch, id_box, ...] / inputshape
                        y_true[an_format][id_batch, grid_two, grid_one, wh_anchor, 4] = 1
                        y_true[an_format][id_batch, grid_two, grid_one, wh_anchor, 5 + class_num] = 1

        return y_true


    def train_data_generator(self, lines, batch_size):
        # plt.ion()
        # fig = plt.figure(figsize = (8,8))
        # ax1 = fig.add_subplot(111)
        anchors = self.read_anchor()
        lines_train = lines
        np.random.shuffle(lines_train)
        num_batch_train = len(lines_train) // batch_size

        for each_batch in range(num_batch_train):

            lines_train_batch = lines_train[batch_size*each_batch: batch_size*(each_batch+1)]

            img_b, box_b, orimg_b = self.parse_annotation(lines_train_batch, batch_size)

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

            yield img_b, true_labels_b, orimg_b

    def validate_data_generator(self, lines, batch_size):
        anchors = self.read_anchor()
        lines_val = lines
        # np.random.shuffle(lines_val)
        num_batch_val = len(lines_val) // batch_size

        selected_batch = 1

        lines_val_batch = lines_val[batch_size*(selected_batch-1): batch_size*selected_batch]

        img_v, box_v, orimage_v = self.parse_annotation(lines_val_batch, batch_size)

        true_labels_v = self.get_train_label(box_v, anchors)

        return img_v, true_labels_v, orimage_v

        
if __name__ == "__main__":
    annotation_file = '/home/azhang/Documents/experiments_in_VIVA/ylt/Yolov3_try_1906/coco_train_self_annotation.txt'
    anchor_boxes_file = '/home/azhang/Documents/experiments_in_VIVA/ylt/Yolov3_try_1906/anchor_kmean_9.txt'
    classes = [0, 1, 2, 3]
    input_shape = (416, 416)
    batch_size_train = 10
    batch_size_val = 100
    num_epoches = 50
    learning_rate_start = 0.0001
    checkp_num = 50
    save_point = 10

    yolo_t = yolo_train(annotation_file, anchor_boxes_file, classes, input_shape)

    lines_train, lines_val = yolo_t.read_and_split_annotations()


    anchors = yolo_t.read_anchor()
    num_classes = len(classes)

    '''
    Test and try implementing my own yolo_v3
    '''
    yolo = YOLO_v3(input_shape, batch_size_train, batch_size_val, anchors, num_classes)

    X = yolo.build_input_holder()
    label = yolo.build_label_holder()
    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(learning_rate_start, global_step, 10000, 0.9, staircase=True)

    num_anchors = len(anchors) // 3

    pred = yolo.build_yolo_body(X)

    # pred = yolo_body(X, num_anchors, num_classes)

    # loss = yolo.loss_function(pred, label)


    loss = yolo.loss_function_third(pred, label)

    tf.summary.scalar('loss_original', tf.squeeze(loss))

    # loss = yolo_loss(pred, label, anchors, num_classes)

    loss_regularized = yolo.filters_regularization(loss, 5e-4)

    tf.summary.scalar('loss_regularized', tf.squeeze(loss_regularized))

    pred_translated = yolo.translate_pred(pred)

    merged = tf.summary.merge_all()
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # variables_unfrozen = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "output_first") + \
        #                     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "output_second") + \
        #                     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "output_third")
        training_op = optimizer.minimize(loss_regularized)

    ##### Implementing average precision calculation in tf ######
    # ap_tf, ap_update = yolo.average_precision_tf(pred_translated, label)
    # running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_acc")
    # running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    ##### INITIALIZATION #####
    init = tf.global_variables_initializer()

    ##### setting disposable memory of GPU #####
    gpu_options = tf.GPUOptions(allow_growth=True)

    ##### save the model ######
    saver = tf.train.Saver()


    '''
    These variables are used for debuging, could be deleted in the fruture.

    '''
    # img_b_f = None
    # label_b_f = None

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # with tf.Session() as sess:
        summaries_dir = 'logs/yolo_test2'
        train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(summaries_dir + '/test')
        summary_for_val = tf.Summary()


        saver.restore(sess, "/home/azhang/Documents/experiments_in_VIVA/Yolov3_try_1906/model/yolo_test_first_third.ckpt")

        batch_counter = 739200

        for epoch in range(num_epoches):

            for imgb, labelb, orimageb in yolo_t.train_data_generator(lines_train, batch_size_train):

                # if img_b_f is None and label_b_f is None:
                #     img_b_f = imgb
                #     label_b_f = labelb
                
                batch_counter += 1

                ''' Training processing '''

                sess.run(training_op, feed_dict = {X: imgb, 
                                                    label[0]: labelb[0],
                                                    label[1]: labelb[1],
                                                    label[2]: labelb[2]})

                loss_current_batch = loss.eval(feed_dict = {X: imgb, 
                                                label[0]: labelb[0],
                                                label[1]: labelb[1],
                                                label[2]: labelb[2]})

                summary_t = sess.run(merged, feed_dict = {X: imgb, 
                                                    label[0]: labelb[0],
                                                    label[1]: labelb[1],
                                                    label[2]: labelb[2]})

                train_writer.add_summary(summary_t, batch_counter)

                if batch_counter % checkp_num == 0:

                    img_v, label_v, orimage_v = yolo_t.validate_data_generator(lines_val, batch_size_val)

                    ''' Validation processing '''

                    pred_v = sess.run(pred_translated, feed_dict = {X: img_v, 
                                                label[0]: label_v[0],
                                                label[1]: label_v[1],
                                                label[2]: label_v[2]})

                    yoloval = yolo_validation(img_v, pred_v, label_v, anchors, input_shape, num_classes)

                    mean_r, mean_p, mean_ap = yoloval.ap_mean_over_batch()

                    summary_for_val.value.add(tag = 'mean_recall', simple_value = mean_r)
                    summary_for_val.value.add(tag = 'mean_precision', simple_value = mean_p)
                    summary_for_val.value.add(tag = 'mAP', simple_value = mean_ap)

                    train_writer.add_summary(summary_for_val, batch_counter)


                    # sess.run(running_vars_initializer)
                    # sess.run(ap_update, feed_dict = {X: img_v, 
                    #                             label[0]: label_v[0],
                    #                             label[1]: label_v[1],
                    #                             label[2]: label_v[2]})

                    # metrics_ap = sess.run(ap_tf)
                    # summary_for_val.value.add(tag = 'mAP', simple_value = metrics_ap)
                    # train_writer.add_summary(summary_for_val, batch_counter)

                    # if batch_counter % save_point == 0:
                    save_path = saver.save(sess, "/home/azhang/Documents/experiments_in_VIVA/Yolov3_try_1906/model/yolo_test_first_fourth.ckpt")


