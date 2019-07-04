# import matplotlib
# matplotlib.use('tkagg')
# import matplotlib.pyplot as plt
import numpy as np
import cv2

np.random.seed(0)

def Kmean_cluster(allinputs, k):
    Threshold = 1e-10

    input_size = allinputs.shape
    Centers_current = np.random.randint(500, size=(k,) + input_size[1:])
    Centers_old = np.random.randint(500, size=(k,) + input_size[1:])

    norm = np.linalg.norm(Centers_current - Centers_old, 'fro')

    while norm > Threshold:
        Centers_old = Centers_current
        Class_belong = np.zeros(input_size[0])
        hm_in_class = np.zeros(k)
        for wh in range(input_size[0]):
            distance = np.zeros(k)
            for ceni in range(k):
                distance[ceni] = np.linalg.norm(allinputs[wh] - Centers_old[ceni])
            belong_to = np.argmin(distance)
            Class_belong[wh] = belong_to
            hm_in_class[belong_to] += 1

        for cenid in range(k):
            sum_up = np.zeros(input_size[1:])
            for wh in range(input_size[0]):
                if Class_belong[wh] == cenid:
                    sum_up += allinputs[wh]
            if hm_in_class[cenid] != 0:
                Centers_current[cenid] = sum_up / hm_in_class[cenid]
            else:
                Centers_current[cenid] = np.random.randint(20, size = input_size[1:])
        
        norm  = np.linalg.norm(Centers_current - Centers_old, 'fro')

    return Centers_current, Class_belong



if __name__ == "__main__":

    All_boxes_wh = []
    input_shape = (416, 416)
    wi, hi = input_shape

    f = open('coco_train_self_annotation.txt', 'r')
    lines = f.readlines()
    for line in lines:
        each_line = line.split(' ')
        image_path = each_line[0]
        image = cv2.imread(image_path)
        ho, wo, channels = image.shape
        scale = np.minimum(hi/ho, wi/wo)
        for i in range(len(each_line)):
            if i == 0: continue
            all_info = np.array(each_line[i].split(',')).astype(np.int32)
            w = all_info[2] * scale
            h = all_info[3] * scale
            All_boxes_wh.append([w, h])
    All_boxes_wh = np.array(All_boxes_wh)
    f.close()

    anchor_number = 9

    centers, wh_class = Kmean_cluster(All_boxes_wh, anchor_number)

    sum_for_sort = np.sum(centers, axis = 1)

    sort_index = np.argsort(sum_for_sort)

    centers = centers[sort_index]

    ##### For saving anchor boxes #####
    f_save = open('anchor_kmean_9.txt', 'w')

    for each_box in range(len(centers)):
        if each_box == 0:
            write_content = "%d,%d" % (centers[each_box][0], centers[each_box][1])
        else:
            write_content = ", %d,%d" % (centers[each_box][0], centers[each_box][1])
        f_save.write(write_content)
    f_save.close()


    ##### For Plotting ######
    # fig = plt.figure(figsize = (10,10))
    # ax1 = fig.add_subplot(111)

    # color_various = ['blue', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'black']

    # ax1.scatter(All_boxes_wh[:,0], All_boxes_wh[:,1], s = 0.03, c = 'blue')

    # ax1.scatter(centers[:,0], centers[:,1], s = 5, c = 'red')

    # plt.show()






