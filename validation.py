import numpy as np
import cv2

class yolo_validation:

    def __init__(self, image, prediction, groundtruth, anchors, inputshape, num_classes):
        self.image = image
        self.prediction = prediction
        self.groundtruth = groundtruth
        self.anchors = anchors
        self.num_format = len(anchors) // 3
        self.inputshape = inputshape
        height, width = inputshape
        self.height = np.float(height)
        self.width = np.float(width)
        self.batch_size = len(self.prediction[0])
        self.num_classes = num_classes

    def transfer_prediction(self):
        '''
        transfer prediction to size (batch_size, total length, 9)
        '''
        all_predictions = []
        for formid in range(self.num_format):
            current_pred = self.prediction[formid]
            current_pred = np.reshape(current_pred, (self.batch_size, -1, 5 + self.num_classes)).astype(np.float32)
            # current_pred[..., :4] *= self.height
            all_predictions.append(current_pred)

        all_predictions = np.concatenate(all_predictions, axis = 1).astype(np.float32)
        # all_predictions[..., :4] *= self.height

        return all_predictions

    def transfer_label(self):
        '''
        transfer ground truth to size (batch_size, total length, 9)
        '''
        all_groundtruth = []
        for formid in range(self.num_format):
            current_gt = self.groundtruth[formid]
            current_gt = np.reshape(current_gt, (self.batch_size, -1, 5 + self.num_classes)).astype(np.float32)
            # current_gt[..., :4] *= self.height
            all_groundtruth.append(current_gt)
        
        all_groundtruth = np.concatenate(all_groundtruth, axis = 1).astype(np.float32)
        # all_groundtruth[..., :4] *= self.height

        output = [None for _ in range(self.batch_size)]
        for batchid in range(self.batch_size):
            current_gt = all_groundtruth[batchid]
            has_obj = current_gt[..., 4] == 1
            current_gt = current_gt[has_obj]
            if len(current_gt) == 0:
                continue
            current_class_num = np.argmax(current_gt[..., 5:], axis = -1).astype(np.float32)
            current_class_num = np.reshape(current_class_num, (len(current_class_num), 1)).astype(np.float32)
            current_gt_formed = np.concatenate((current_gt[:, :5], current_class_num), axis = -1)
            
            output[batchid] = current_gt_formed

        return output

    def bbox_iou(self, box1, box2, x1y1x2y2=False):
        """
        Returns the IoU of two bounding boxes
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # get the corrdinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1).astype(np.float32)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1).astype(np.float32)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2).astype(np.float32)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2).astype(np.float32)
        # Intersection area
        inter_area = np.maximum(inter_rect_x2 - inter_rect_x1 + 1, 0).astype(np.float32) * np.maximum(inter_rect_y2 - inter_rect_y1 + 1, 0).astype(np.float32)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def box_iou(self, b1, b2):
        '''
        b1 <- (1, 4) <- (xcen, ycen, w, h)
        b2 <- (batch_num, 4) <- (xcen, ycen, w, h)
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


    def NMS(self, objthreshold = 0.4, nmsthreshold = 0.6):
        '''
        Self Implementation of Non-maximum Suppression
        '''
        all_predictions = self.transfer_prediction()
        output = [None for _ in range(len(all_predictions))]
        for batid in range(self.batch_size):
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
            # print(pred_new[:10, -1])

            box_output = []
            while (len(pred_new) != 0):
                fisrt_box = pred_new[0, :4]
                fisrt_box = np.expand_dims(fisrt_box, axis = 0).astype(np.float32)
                iou_inside = self.box_iou(fisrt_box, pred_new[:, :4])
                # print(iou_inside[0,...])
                # print(iou_inside)
                invalid_overlap = iou_inside >= nmsthreshold
                same_class = pred_new[0, -1] == pred_new[:, -1]
                discarded_boxes = invalid_overlap & same_class
                
                discarded_objness = pred_new[discarded_boxes, 4:5]
                # if np.sum(discarded_objness) != 0:
                # pred_new[0, :4] = np.sum(discarded_objness * pred_new[discarded_boxes, :4], axis = 0) / np.sum(discarded_objness)
                box_output.append(pred_new[0])
                pred_new = pred_new[~discarded_boxes]
            
            if len(box_output) != 0:
                box_output = np.stack(box_output)
                output[batid] = box_output

        return output
    
    def get_true_positive(self, iou_threshold = 0.5):
        '''
        Get the true positive boxes from comparing predcitions with ground truth
        '''
        pred_nms = self.NMS()
        true_label = self.transfer_label()

        batch_metrics = []

        for bathid in range(self.batch_size):

            if pred_nms[bathid] is None:
                continue
            
            pred_b = pred_nms[bathid]
            pred_num_box = len(pred_b)
            # print(pred_num_box)
            pred_b_boxes = pred_b[:, :4]
            pred_b_scores = pred_b[:, 4]
            pred_b_class = pred_b[:, -1]
            # print([len(pred_b_scores), len(pred_b_class)])

            true_b = true_label[bathid]

            true_positive = np.zeros(pred_num_box)
            if true_b is not None:
                true_class = true_b[:, -1]
                true_boxes = true_b[:, :4]
                true_scores = true_b[:, 4]

                detected_boxes = []

                for pnbox in range(pred_num_box):
                    pred_each_class = pred_b_class[pnbox]
                    pred_each_box = pred_b_boxes[pnbox]
                    pred_each_box = np.expand_dims(pred_each_box, axis = 0)

                    if len(detected_boxes) == len(true_b): break
                    if pred_each_class not in true_class: continue
                    
                    true_prdcla_index = np.where(true_class == pred_each_class)[0]

                    true_blong_to_pred_class = true_class == pred_each_class
                    true_box_predclass = true_boxes[true_blong_to_pred_class]

                    iou_all = self.box_iou(pred_each_box, true_box_predclass)
                    iou_max = np.amax(iou_all, axis = 0)
                    iou_max_ind = np.argmax(iou_all, axis = 0)

                    true_index = true_prdcla_index[iou_max_ind]
                    # print([iou_max ,iou_max_ind])
                    if iou_max >= iou_threshold and true_index not in detected_boxes:
                        true_positive[pnbox] = 1
                        detected_boxes.append(true_index)
                # print(len(detected_boxes))
            batch_metrics.append([true_positive, pred_b_scores, pred_b_class, true_b])

        return batch_metrics

    def compute_ap(self, recall, precision):
        ''' 
        Compute the average precision, given the recall and precision curves.
        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        '''
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap

    def ap_over_class(self, true_positive, pred_score, pred_class, true_label):
        '''
        Computing the average precision.
        '''
        score_order = np.argsort( - pred_score )
        true_classes = true_label[:, -1]
        true_positive, pred_score, pred_class = true_positive[score_order], pred_score[score_order], pred_class[score_order]

        true_unique_class = np.unique(true_classes)

        total_num_calsses = len(true_unique_class)

        ap, precision, recall = [], [], []

        precision_sum = 0
        recall_sum = 0
        ap_sum = 0

        for c in true_unique_class:
            pred_is_this_class = pred_class == c
            turth_is_this_class = true_classes == c
            num_gt = len(true_classes[turth_is_this_class])
            num_prediciton = len(pred_class[pred_is_this_class])
            if num_prediciton == 0 and num_gt == 0:
                continue
            elif num_prediciton == 0 or num_gt == 0:
                ap.append(0)
                precision.append(0)
                recall.append(0)
            else:
                ##### Compute the number of ture positive and the number of false positive
                num_tp = np.cumsum(true_positive[pred_is_this_class])
                num_fp = np.cumsum(1 - true_positive[pred_is_this_class])

                ##### Recall #####
                recall_class = num_tp / (num_gt + 1e-16)
                recall.append(recall_class[-1])
                recall_sum += recall_class[-1]

                ##### Precision #####
                precision_class = num_tp / (num_tp + num_fp)
                precision.append(precision_class[-1])
                precision_sum += precision_class[-1]
                
                ##### AP from recall-precision curve #####
                ap_class = self.compute_ap(recall_class, precision_class)
                ap.append(ap_class)
                ap_sum += ap_class
            
        mean_r = recall_sum / total_num_calsses
        mean_p = precision_sum / total_num_calsses
        mean_ap = ap_sum / total_num_calsses

        return mean_r, mean_p, mean_ap
        

    def ap_mean_over_batch(self):
        '''
        Average the AP over one validation batch.
        '''
        gt_tttt = self.transfer_label()
        gt_num = 0
        for iid in range(self.batch_size):
            if gt_tttt[iid] is not None:
                gt_num += len(gt_tttt[iid])

        all_pred = self.NMS()
        total_num = 0
        for bbid in range(self.batch_size):
            if all_pred[bbid] is not None:
                total_num += len(all_pred[bbid])

        batch_metrics = self.get_true_positive()

        tp_all = []
        pred_sc_all = []
        pred_cl_all = []
        true_all = []
        if len(batch_metrics) != 0:
            for metrid in range(len(batch_metrics)):
                tp, pred_sc, pred_cl, true = batch_metrics[metrid]
                tp_all.append(tp)
                pred_sc_all.append(pred_sc)
                pred_cl_all.append(pred_cl)
                true_all.append(true)
            tp_all = np.concatenate(tp_all, axis = 0)
            pred_sc_all = np.concatenate(pred_sc_all, axis = 0)
            pred_cl_all = np.concatenate(pred_cl_all, axis = 0)
            true_all = np.concatenate(true_all, axis = 0)

            # print("print for test")
            # print(tp_all.shape)
            # print(pred_sc_all.shape)
            # print(pred_cl_all.shape)
            # print(true_all.shape)

            mean_r, mean_p, mean_ap = self.ap_over_class(tp_all, pred_sc_all, pred_cl_all, true_all)
        else: mean_r, mean_p, mean_ap = 0.0, 0.0, 0.0

        return mean_r, mean_p, mean_ap

