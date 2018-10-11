import os

labels_file = 'imagenet_labels.txt'
results_file = 'result/grad_result_adv_rar_iou.txt'
# if os.path.exists(results_file):
#     os.remove(results_file)
rar_iou = 0
adv_iou = 0
s=0
a=0
b=0
with open(results_file, 'r', encoding='utf8')as f:
    lines = f.readlines()
    lenth = len(lines)
    for line in lines:
        label_letter = line.split(' ')
        if label_letter[1] == 'True':
            rar_iou += float(label_letter[4])
            a+=1
        else:
            adv_iou += float(label_letter[4])
            b+=1
        s+= float(label_letter[4])
    print(rar_iou / a, adv_iou / b,s/lenth)
fp = open('ILSVRC2012_val_00000293.xml')
from  matplotlib import pyplot as plt
import numpy as np


def get_gound_truth():
    ground_truth = np.zeros((299, 299))
    for p in fp:
        if '<size>' in p:
            width = int(next(fp).split('>')[1].split('<')[0])
            height = int(next(fp).split('>')[1].split('<')[0])
        if '<object>' in p:
            print(next(fp).split('>')[1].split('<')[0])
        if '<bndbox>' in p:
            xmin = int(next(fp).split('>')[1].split('<')[0])
            ymin = int(next(fp).split('>')[1].split('<')[0])
            xmax = int(next(fp).split('>')[1].split('<')[0])
            ymax = int(next(fp).split('>')[1].split('<')[0])
            matrix = [int(xmin / width * 299), int(ymin / height * 299), int(xmax / height * 299),
                      int(ymax / height * 299)]
            ground_truth[matrix[0]:matrix[2], matrix[1]:matrix[3]] = 1
    return ground_truth


import matplotlib.pyplot as plt
