import os

import cv2
import numpy as np
from skimage.transform import resize

import matplotlib.pyplot as plt

key = 10


def make_cam(x):
    x = np.reshape(x, (8, 8))
    threshold = np.sort(np.reshape(x, (64)))[-int(key)]
    x = x - threshold
    x = resize(x, [299, 299])
    x[x < 0] = 0
    x[x > 0] = 1
    return x


def show_img(img, rar_mask, adv_mask, gt):
    plt.figure()
    # rar_mask=make_cam(rar_mask)
    rar_mask=np.resize(rar_mask,(299,299))
    rar_mask = resize(rar_mask, (299, 299))
    plt.subplot(1, 4, 1)
    plt.imshow(rar_mask)
    # img = cv2.resize(img, (299, 299))
    # img = img.astype(float)
    # img /= img.max()
    # rar_mask = cv2.applyColorMap(np.uint8(255 * rar_mask), cv2.COLORMAP_JET)
    # rar_mask = cv2.cvtColor(rar_mask, cv2.COLOR_BGR2RGB)
    # alpha = 0.0072
    # rar_img = img + alpha * rar_mask
    # rar_img = rar_img / np.max(rar_img)
    # adv_mask = cv2.resize(rar_mask, (299, 299))
    # adv_mask = cv2.applyColorMap(np.uint8(255 * adv_mask), cv2.COLORMAP_JET)
    # adv_mask = cv2.cvtColor(adv_mask, cv2.COLOR_BGR2RGB)
    # adv_img = img + alpha * adv_mask
    #
    # # plt.subplot(1, 4, 1)
    # # plt.imshow(img)
    # plt.subplot(1, 4, 2)
    # plt.imshow(rar_mask)
    #
    # plt.subplot(1, 4, 3)
    # plt.imshow(rar_img)
    #
    # plt.subplot(1, 4, 4)
    # gt = cv2.applyColorMap(np.uint8(255 * gt), cv2.COLORMAP_JET)
    # gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    # gt_img = img + alpha * gt
    #
    # plt.imshow(gt_img)
    plt.show()


if __name__ == '__main__':
    results_file = str(key) + 'lime_iou.txt'
    temp_rar = 0
    temp_adv = 0
    defense_rar_gt_iou = 0
    defense_adv_gt_iou = 0
    attack_adv_gt_iou = 0
    attack_rar_gt_iou = 0
    defense_iou = 0
    attack_iou = 0
    defense_count = 0
    attack_count = 0
    total_num = 0
    if os.path.isfile(results_file):
        os.remove(results_file)
    for root, dirs, files in os.walk('npz'):
        for i, f in enumerate(files):
            try:
                arr = np.load("npz/" + f)
            except Exception as e:
                print(e)
                continue
            print(f)
            filename = f.split('_')
            pred_label = filename[0]
            adv_label = filename[1]
            img = arr['arr_0']
            rar = arr['arr_1']
            adv = arr['arr_2']
            gt = arr['arr_3']
            show_img(img, rar, adv, gt)
