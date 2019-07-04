import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import skimage.io as io
import time

plt.ion()

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)


f = open('coco_train_self_annotation_vehicle.txt', 'r')

lines = f.readlines()

# print(np.array(lines[0].split(' ')[0].split(',')).astype(np.int32))

for line in lines:
    each_line = line.split(' ')
    img_location = each_line[0]
    image = cv2.imread(img_location)
    plt.cla()
    ax.clear()
    ax.imshow(image)
    for i in range(len(each_line)):
        if i == 0: continue
        all_info = np.array(each_line[i].split(',')).astype(np.int32)
        x = all_info[0]
        y = all_info[1]
        w = all_info[2]
        h = all_info[3]
        box_class = all_info[4]
        if box_class == 1:
            rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='blue',facecolor='none')
        elif box_class == 3:
            rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='red',facecolor='none')
        elif box_class == 8:
            rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='yellow',facecolor='none')
        ax.add_patch(rect)
    fig.canvas.draw()
    plt.show()
    plt.pause(3)

