import tensorflow as tf
import numpy as np

class YOLO_v3:
    def __init__(self, input_shape, batch_size, batch_size_val, anchor, num_classes):
        if len(input_shape) == 2:
            self.height = input_shape[0]
            self.width = input_shape[1]
            self.channels = 3
        elif len(input_shape) == 3:
            self.height = input_shape[0]
            self.width = input_shape[1]
            self.channels = input_shape[2]
        else: 
            print("Please get right input size.")
            self.height = 0
            self.width = 0 
            self.channels = 0
        self.batch_size = batch_size
        self.anchors = anchor
        self.num_anchor = len(anchor) // 3
        self.num_classes = num_classes
        self.avoid_inf = 1e-10
        self.batch_size_val = batch_size_val

    def build_input_holder(self):
        INPUT = tf.placeholder(tf.float32, shape = (None, self.height, self.width, self.channels))
        return INPUT

    def build_label_holder(self):
        LABEL1 = tf.placeholder(tf.float32, shape = (None, self.height//32, self.width//32, self.num_anchor, 5 + self.num_classes))
        LABEL2 = tf.placeholder(tf.float32, shape = (None, self.height//16, self.width//16, self.num_anchor, 5 + self.num_classes))
        LABEL3 = tf.placeholder(tf.float32, shape = (None, self.height//8, self.width//8, self.num_anchor, 5 + self.num_classes))
        return [LABEL1, LABEL2, LABEL3]
    
    def conv_head(self, input_data):
        filters = tf.Variable(tf.glorot_uniform_initializer()([3, 3, 3, 32]))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, filters)
        padding = "SAME"
        strides = [1, 1, 1, 1]
        header_layer = tf.nn.conv2d(input_data, filters, strides = strides, padding = padding)
        header_layer = tf.keras.layers.BatchNormalization()(header_layer)
        header_layer = tf.nn.leaky_relu(header_layer, alpha = 0.1)
        return header_layer

    def block_iteration(self, input_data, kernel_size, input_nfilters, output_nfilters):
        w, h = kernel_size
        filters = tf.Variable(tf.glorot_uniform_initializer()([w, h, input_nfilters, output_nfilters]))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, filters)
        padding = "SAME"
        strides = [1, 1, 1, 1]
        blockbody = tf.nn.conv2d(input_data, filters, strides = strides, padding = padding)
        blockbody = tf.keras.layers.BatchNormalization()(blockbody)
        blockbody = tf.nn.leaky_relu(blockbody, alpha = 0.1)
        return blockbody

    def connector_and_block(self, input_data, input_nfilters, output_nfilters, num_iteration):
        filters = tf.Variable(tf.glorot_uniform_initializer()([3, 3, input_nfilters, output_nfilters]))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, filters)
        top_left_pad = tf.constant([[0, 0], [1, 0], [1, 0], [0, 0]])
        input_tlpad = tf.pad(input_data, top_left_pad, "CONSTANT")
        padding = "VALID"
        strides = [1, 2, 2, 1]
        connector = tf.nn.conv2d(input_tlpad, filters, strides = strides, padding = padding)
        connector = tf.keras.layers.BatchNormalization()(connector)
        connector = tf.nn.leaky_relu(connector, alpha = 0.1)
        additional_data = connector
        output_data = connector
        with tf.name_scope("blockbody"):
            for ite_i in range(num_iteration):
                output_data = self.block_iteration(output_data, (1,1), output_nfilters, input_nfilters)
                output_data = self.block_iteration(output_data, (3,3), input_nfilters, output_nfilters)
                additional_data = tf.add(output_data, additional_data)
                output_data = additional_data
        return output_data

    def last_layers(self, input_data, input_nfilters, output_nfilters, final_nfilters):
        out_first = self.block_iteration(input_data, (1,1), input_nfilters, output_nfilters)
        out_first = self.block_iteration(out_first, (3,3), output_nfilters, input_nfilters)
        out_first = self.block_iteration(out_first, (1,1), input_nfilters, output_nfilters)
        out_first = self.block_iteration(out_first, (3,3), output_nfilters, input_nfilters)
        out_first = self.block_iteration(out_first, (1,1), input_nfilters, output_nfilters)

        out_second = self.block_iteration(out_first, (3,3), output_nfilters, input_nfilters)
        out_second = self.block_iteration(out_second, (1,1), input_nfilters, final_nfilters)

        return out_first, out_second

    def build_yolo_body(self, input_data):
        '''
        buiding structure
        '''
        with tf.name_scope("header"):
            header = self.conv_head(input_data)
        with tf.name_scope("bolck_one"):
            block1 = self.connector_and_block(header, 32, 64, 1)
        with tf.name_scope("block_two"):
            block2 = self.connector_and_block(block1, 64, 128, 2)
        with tf.name_scope("block_three"):
            ### layer for further upsampling use ###
            block3 = self.connector_and_block(block2, 128, 256, 8)
        with tf.name_scope("block_four"):
            ### layer for further upsampling use ###
            block4 = self.connector_and_block(block3, 256, 512, 8)
        with tf.name_scope("block_five"):
            block5 = self.connector_and_block(block4, 512, 1024, 4)

        '''
        prepare the output for the first group of grids
        '''
        final_size = self.num_anchor * (5 + self.num_classes)
        with tf.name_scope("output_first"):
            feature_map_first, pred_first = self.last_layers(block5, 1024, 512, final_size)
        
        with tf.name_scope("output_second"):
            concat_input_prefirst = self.block_iteration(feature_map_first, (1,1), 512, 256)
            ##### keras high-end upsampling #####
            concat_input_first = tf.keras.layers.UpSampling2D(2)(concat_input_prefirst)
            second_input = tf.concat([concat_input_first, block4], axis = -1)
            feature_map_second, pred_second = self.last_layers(second_input, 768, 256, final_size)

        with tf.name_scope("output_third"):
            concat_input_presecond = self.block_iteration(feature_map_second,  (1,1), 256, 128)
            concat_input_second = tf.keras.layers.UpSampling2D(2)(concat_input_presecond)
            third_input = tf.concat([concat_input_second, block3], axis = -1)
            feature_map_third, pred_third = self.last_layers(third_input, 384, 128, final_size)

        return [pred_first, pred_second, pred_third]

    def translate_label(self, pred_spec, anchor_spec, input_shape):
        '''
        Transfer the result with shape (3, 9) to bounding box and confidence.
        '''
        hm_anchors = len(anchor_spec)
        num_batch = tf.shape(pred_spec)[0]
        anchors_tensor = tf.reshape(tf.constant(anchor_spec, dtype = pred_spec.dtype), [1, 1, 1, hm_anchors, 2])
        grid_size = tf.shape(pred_spec)[1:3]
        pred_stdform = tf.reshape(pred_spec, [-1, grid_size[0], grid_size[1], self.num_anchor, 5 + self.num_classes])

        ##### get the xycenter and wh for calculating loss function #####
        pred_xycenter = pred_stdform[..., 0:2]
        pred_wh = pred_stdform[..., 2:4]

        ##### get the objectness and classes for evaluation #####
        pred_objectness = pred_stdform[..., 4:5]
        pred_objectness = tf.sigmoid(pred_objectness)
        pred_classes = pred_stdform[..., 5:]
        pred_classes = tf.sigmoid(pred_classes)

        ##### get the prediction bounding boxes for evaluation #####
        grid_w = tf.tile(tf.reshape(tf.range(0, grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_h = tf.tile(tf.reshape(tf.range(0, grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.cast(tf.concat([grid_h, grid_w], axis = -1), dtype = pred_spec.dtype)

        pred_box_xy = (tf.sigmoid(pred_xycenter) + grid) / tf.cast(grid_size, pred_spec.dtype)
        pred_box_wh = tf.exp(pred_wh) * anchors_tensor / tf.cast(input_shape, dtype = pred_spec.dtype)
        # pred_box_xystart = pred_box_xy - pred_box_wh / 2
        pred_bbox = tf.concat([pred_box_xy, pred_box_wh], axis = -1)

        return grid, pred_bbox, pred_objectness, pred_classes, pred_stdform

    def calculate_iou(self, box1, box2):
        '''
        box1 <- (grid_size1, grid_size2, 3, 4)
        box2 <- (num_trueboxes, 4)

        iou_output <- (grid_size1, grid_size2, 3, num_trueboxes)
        '''
        ##### Using tf broadcasting to calculate element-wise calcultions #####
        box1 = tf.expand_dims(box1, -2)
        box1_xy = box1[..., :2]
        box1_wh = box1[..., 2:4]
        box1_min = box1_xy - box1_wh / 2.
        box1_max = box1_xy + box1_wh / 2.

        box2 = tf.expand_dims(box2, 0)
        box2_xy = box2[..., :2]
        box2_wh = box2[..., 2:4]
        box2_min = box2_xy - box2_wh / 2.
        box2_max = box2_xy + box2_wh / 2.

        intersection_min = tf.maximum(box1_min, box2_min)
        intersection_max = tf.minimum(box1_max, box2_max)
        intersection_wh = tf.maximum(intersection_max - intersection_min, 0.)
        intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
        box1_area = box1_wh[..., 0] * box1_wh[..., 1]
        box2_area = box2_wh[..., 0] * box2_wh[..., 1]
        iou = intersection / (box1_area + box2_area - intersection)

        return iou
    
    def find_ignore_masks(self, predbox, ture_xywh, object_mask_bool, iou_threshold):
        '''
        Set a iou threshold to identify the boxes.
        '''
        output_shape = tf.shape(predbox)
        nonobj_mask = []
        for each in range(self.batch_size):
            object_true_box = tf.boolean_mask(ture_xywh[each], object_mask_bool[each])
            each_iou = self.calculate_iou(predbox[each], object_true_box)
            best_iou = tf.reduce_max(each_iou, axis = -1)
            ignore_mask = tf.cast(best_iou < iou_threshold, dtype = predbox[each].dtype)
            nonobj_mask.append(ignore_mask)
        
        nonobj_mask = tf.stack(nonobj_mask)

        return nonobj_mask

    # def loss_function(self, pred, label_true, iou_threshold = 0.5):
    #     '''
    #     Define loss function according to the yolov1 paper
    #     '''
    #     num_grid_format = self.num_anchor
    #     anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_grid_format==3 else [[3,4,5], [1,2,3]]
    #     input_shape_tensor = tf.constant([self.height, self.width], dtype = label_true[0].dtype)
    #     grid_shape = [tf.cast(tf.shape(pred[j])[1:3], dtype = label_true[0].dtype) for j in range(num_grid_format)]
    #     batch_size_float = tf.constant([self.batch_size], dtype = label_true[0].dtype)
    #     loss = 0
    #     for format_i in range(num_grid_format):
    #         ture_xywh = label_true[format_i][..., 0:4]
    #         objectness_true = label_true[format_i][..., 4:5]
    #         objectness_true_bool = tf.cast(objectness_true, tf.bool)
    #         ##### find those who under the iou threshold #####
    #         object_mask = objectness_true
    #         object_mask_forcal = tf.squeeze(objectness_true, -1)
    #         object_mask_bool = tf.cast(object_mask_forcal, tf.bool)
            
    #         classes_true = label_true[format_i][..., 5:]
    #         ##### get prediction xy, wh for training #####
    #         grid, pred_bbox, pred_objectness, pred_classes, pred_xywh = self.translate_label(pred[format_i], self.anchors[anchor_mask[format_i]], input_shape_tensor)

    #         pred_xy = pred_xywh[..., 0:2]
    #         pred_wh = pred_xywh[..., 2:4]
    #         ##### get true label xy, wh for training #####
    #         xy_true = label_true[format_i][..., :2] * grid_shape[format_i] - grid
    #         wh_true = tf.math.log(label_true[format_i][..., 2:4]  / self.anchors[anchor_mask[format_i]] * input_shape_tensor)
    #         ##### avoid infinite #####
    #         wh_true = tf.where(tf.is_inf(wh_true),tf.zeros_like(wh_true), wh_true)

    #         # wh_true = tf.keras.backend.switch(objectness_true_bool, wh_true, tf.zeros_like(wh_true))

    #         ##### give larger weights to small boxes #####
    #         box_loss_scale = 2 - label_true[format_i][..., 2:3] * label_true[format_i][..., 3:4]

    #         nonobj_mask = self.find_ignore_masks(pred_bbox, ture_xywh, object_mask_bool, iou_threshold)
    #         nonobj_mask = tf.expand_dims(nonobj_mask, -1)

    #         # ##### CALCULATING ALL LOSS FUNCTIONS #####
    #         xy_loss = object_mask * box_loss_scale * tf.keras.backend.binary_crossentropy(xy_true - pred_xy, from_logits=True)
    #         wh_loss = object_mask * box_loss_scale * tf.square(wh_true - pred_wh)

    #         # object_loss = tf.keras.backend.binary_crossentropy(object_mask, pred_xywh[..., 4:5], from_logits=True) # NO SIGMOID
    #         # pred_objectness = tf.expand_dims(pred_objectness, axis = -1)
    #         object_loss = tf.keras.backend.binary_crossentropy(object_mask, pred_objectness, from_logits=True) # Using Sigmoid before training
    #         confidence_loss = object_mask * object_loss + (1 - object_mask) * nonobj_mask * object_loss
            
    #         # class_loss = tf.keras.backend.binary_crossentropy(classes_true, pred_xywh[..., 5:], from_logits=True) # NO SIGMOID
    #         class_loss = tf.keras.backend.binary_crossentropy(classes_true, pred_classes, from_logits=True) # Using Sigmoid before training
    #         class_loss = objectness_true * class_loss

    #         ##### normalize the loss functions #####
    #         xy_loss = tf.reduce_sum(xy_loss) / batch_size_float
    #         wh_loss = tf.reduce_sum(wh_loss) / batch_size_float
    #         confidence_loss = tf.reduce_sum(confidence_loss) / batch_size_float
    #         class_loss = tf.reduce_sum(class_loss) / batch_size_float

    #         ##### sum all loss functions together to create final loss function for training #####
    #         loss += xy_loss + wh_loss + confidence_loss + class_loss

    #     return loss


    def loss_function_third(self, pred, label_true, ignore_thresh = 0.5, object_scale = 1, noobject_scale = 1):
        '''
        Define loss function according to the yolov1 paper
        '''
        num_grid_format = self.num_anchor
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_grid_format==3 else [[3,4,5], [1,2,3]]
        input_shape_tensor = tf.constant([self.height, self.width], dtype = label_true[0].dtype)
        grid_shape = [tf.cast(tf.shape(pred[j])[1:3], dtype = label_true[0].dtype) for j in range(num_grid_format)]
        batch_size_float = tf.constant([self.batch_size], dtype = label_true[0].dtype)
        loss = 0
        for format_i in range(num_grid_format):
            ture_xywh = label_true[format_i][..., 0:4]
            object_mask = label_true[format_i][..., 4:5]
  
            classes_true = label_true[format_i][..., 5:]
            ##### get prediction xy, wh for training #####
            grid, pred_bbox, pred_objectness, pred_classes, pred_stdform = self.translate_label(pred[format_i], self.anchors[anchor_mask[format_i]], input_shape_tensor)

            pred_xy = pred_stdform[..., 0:2]
            pred_wh = pred_stdform[..., 2:4]

            ##### get true label xy, wh for training #####
            xy_true_transfer = label_true[format_i][..., :2] * grid_shape[format_i] - grid
            wh_true_transfer = tf.math.log(label_true[format_i][..., 2:4]  / self.anchors[anchor_mask[format_i]]  * input_shape_tensor + 1e-16)

            # wh_true_transfer = label_true[format_i][..., 2:4]  / self.anchors[anchor_mask[format_i]] * input_shape_tensor

            ##### avoid infinite #####
            wh_true_transfer = tf.where(tf.is_inf(wh_true_transfer),tf.zeros_like(wh_true_transfer), wh_true_transfer)

            ##### give larger weights to small boxes #####
            box_loss_scale = 2 - label_true[format_i][..., 2:3] * label_true[format_i][..., 3:4]

            ##### Calculate ignore mask #####
            ignore_mask = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
            object_mask_bool = tf.cast(object_mask, tf.bool)
            def loop_body(b, ignore_mask):
                true_box = tf.boolean_mask(label_true[format_i][b,...,0:4], object_mask_bool[b,...,0])
                iou = self.calculate_iou(pred_bbox[b], true_box) 
                best_iou = tf.reduce_max(iou, axis=-1)
                ignore_mask = ignore_mask.write(b, tf.cast(best_iou<ignore_thresh, tf.float32))
                return b+1, ignore_mask
            _, ignore_mask = tf.while_loop(lambda b,*args: b<self.batch_size, loop_body, [0, ignore_mask])
            ignore_mask = ignore_mask.stack()
            ignore_mask = tf.expand_dims(ignore_mask, axis=-1)
            
            noobj_mask = object_mask * object_scale + (1 - object_mask) * ignore_mask * noobject_scale

            ##### Calculate each loss #####
            xy_loss = object_mask * box_loss_scale * tf.square(xy_true_transfer - tf.sigmoid(pred_stdform[..., 0:2]))
            wh_loss = object_mask * box_loss_scale * tf.square(wh_true_transfer - pred_stdform[..., 2:4])
            conf_loss = noobj_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels = object_mask, logits = pred_stdform[..., 4:5])
            class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels = classes_true, logits = pred_stdform[..., 5:])

            ##### normalize the loss functions #####
            xy_loss_sum = tf.reduce_mean(xy_loss)
            wh_loss_sum = tf.reduce_mean(wh_loss)
            conf_loss_sum = tf.reduce_sum(conf_loss) / batch_size_float
            class_loss_sum = tf.reduce_sum(class_loss) / batch_size_float

            ##### sum all loss functions together to create final loss function for training #####
            loss = loss + xy_loss_sum + wh_loss_sum + conf_loss_sum + class_loss_sum

        return loss


    def filters_regularization(self, loss, regularization_lambda):
        regularization_return = 0
        all_filters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for i in range(len(all_filters)):
            regularization_return += regularization_lambda * tf.reduce_sum(tf.square(all_filters[i]))
        regularization_return = regularization_return + loss

        return regularization_return

    def translate_pred(self, pred):
        num_grid_format = self.num_anchor
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_grid_format==3 else [[3,4,5], [1,2,3]]
        input_shape_tensor = tf.constant([self.height, self.width], dtype = pred[0].dtype)
        input_bsize = tf.shape(pred[0])[0]
        output = []
        for formatid in range(num_grid_format):
            ##### get prediction xy, wh for training #####
            grid, pred_bbox, pred_objectness, pred_classes, pred_stdform = self.translate_label(pred[formatid], self.anchors[anchor_mask[formatid]], input_shape_tensor)
            grid_size = tf.shape(pred[formatid])[1:3]
            # pred_objectness = tf.expand_dims(pred_objectness, axis = -1)
            output_single1 = tf.concat([pred_bbox, pred_objectness, pred_classes], axis = -1)
            output_single = tf.reshape(output_single1, [-1, grid_size[0], grid_size[1], self.num_anchor, 5 + self.num_classes])
            output.append(output_single)

        return output

    def average_precision_tf(self, pred, label):
        '''
        Use tensorflow built-in function to calculate average precision, for better gpu memory management.
        '''
        batch_size = tf.shape(pred[0])[0]
        all_pred = []
        all_label = []
        num_grid_format = self.num_anchor
        for foid in range(num_grid_format):
            current_pred = pred[foid]
            current_label = label[foid]

            current_pred = tf.cast(tf.reshape(current_pred, [batch_size, -1, 5 + self.num_classes]), dtype = tf.float32)
            current_label = tf.cast(tf.reshape(current_label, [batch_size, -1, 5 + self.num_classes]), dtype = tf.int64)
            
            all_pred.append(current_pred)
            all_label.append(current_label)
        
        all_pred = tf.concat(all_pred, axis = 1)
        all_label = tf.concat(all_label, axis = 1)

        tf_metric, tf_metric_update = tf.metrics.average_precision_at_k(all_label, all_pred, 3, name = "my_acc")

        return tf_metric, tf_metric_update

    '''
    The following functions are used for reading the predicitons and applying NMS on GPU.
    Specifically, the output could be directly used in picture prediciton.
    '''   
    
    def yolo_output_box_translation(self, pred, image_shape):
        num_grid_format = self.num_anchor
        img_h, img_w = image_shape
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_grid_format==3 else [[3,4,5], [1,2,3]]
        input_shape_tensor = tf.constant([self.height, self.width], dtype = pred[0].dtype)
        image_shape_tensor = tf.constant([img_h, img_w], dtype = pred[0].dtype)
        input_bsize = tf.shape(pred[0])[0]
        boxes = []
        conf_scores = []
        for formatid in range(num_grid_format):
            ##### get prediction xy, wh for training #####
            grid, pred_bbox, pred_objectness, pred_classes, pred_stdform = self.translate_label(pred[formatid], self.anchors[anchor_mask[formatid]], input_shape_tensor)
            ##### scaled the box back to the size of image #####
            scale = tf.reduce_min(input_shape_tensor / image_shape_tensor)
            resized_shape = tf.round(image_shape_tensor * scale)
            offset_flip = (image_shape_tensor - resized_shape) / 2.
            box_xycen = pred_bbox[..., 0:2] * input_shape_tensor - offset_flip[::-1]
            box_wh = pred_bbox[..., 2:4] * input_shape_tensor
            box_xymin = box_xycen - ( box_wh / 2. )
            box_xymax = box_xycen + ( box_wh / 2. )
            box_xymin = tf.round(box_xymin / scale)
            box_xymax = tf.round(box_xymax / scale)
            # box_wh = tf.round(box_wh / scale)

            target_box = tf.concat([box_xymin, box_xymax], axis = -1)

            target_box = tf.reshape(target_box, [-1, 4])
            boxes.append(target_box)

            ##### get the box scores #####
            # pred_objectness = tf.expand_dims(pred_objectness, axis=-1)
            box_scores = pred_objectness * pred_classes
            box_scores = tf.reshape(box_scores, [-1, self.num_classes])
            conf_scores.append(box_scores)
        
        boxes = tf.concat(boxes, axis = 0)
        conf_scores = tf.concat(conf_scores, axis = 0)
        
        return boxes, conf_scores

    def NMS_pred(self, pred, image_shape, max_boxes = 20, score_threshold = 0.6, iou_threshold = 0.5):
        '''
        Get the unique output of yolo detection.
        '''
        boxes, bscores = self.yolo_output_box_translation(pred, image_shape)
        score_filter_mask = bscores >= score_threshold
        max_boxes_tensor = tf.constant(max_boxes, dtype = tf.int32)
        output_boxes = []
        output_scores = []
        output_classes = []
        for c in range(self.num_classes):
            class_boxes = tf.boolean_mask(boxes, score_filter_mask[:,c])
            class_bscore = tf.boolean_mask(bscores[:,c], score_filter_mask[:,c])
            nms_index = tf.image.non_max_suppression(class_boxes, class_bscore, max_boxes_tensor, iou_threshold=iou_threshold)

            selected_boxes = tf.gather(class_boxes, nms_index)
            selected_bscores = tf.gather(class_bscore, nms_index)
            selected_classes = tf.ones_like(selected_bscores, tf.int32) * c
            output_boxes.append(selected_boxes)
            output_scores.append(selected_bscores)
            output_classes.append(selected_classes)
        output_boxes = tf.concat(output_boxes, axis = 0)
        output_scores = tf.concat(output_scores, axis = 0)
        output_classes = tf.concat(output_classes, axis = 0)

        return output_boxes, output_scores, output_classes






if __name__ == "__main__":
    input_shape = (416, 416)
    batch_size = 32 
    yolo = YOLO_v3(input_shape, batch_size)
    yolo.yolo_body_construct()
