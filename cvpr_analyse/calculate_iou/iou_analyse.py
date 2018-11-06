import os
import matplotlib.pyplot as plt
import numpy as np


def get_rar_gt_iou(rar,adv, ground_truth):
    ground_count=ground_truth[ground_truth==1].size
    rar_sum = rar + ground_truth
    # adv_sum = adv + ground_truth
    rar_IOU = rar_sum[rar_sum == 2].size / ground_count
    return rar_IOU


def get_rar_adv_iou(rar, adv):
    sum = rar + adv
    IOU = sum[sum == 2].size / sum[sum != 0].size
    return IOU

from skimage.transform import resize
key=8
def make_cam(x):
    threshold = np.sort(np.reshape(x, (64)))[-int(key)]
    x = x - threshold
    x = resize(x, [299, 299])
    x[x < 0] = 0
    x[x > 0] = 1
    return x
if __name__ == '__main__':
    results_file = 'lime_iou.txt'
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
            filename = f.split('_')
            pred_label = filename[0]
            adv_label = filename[1]
            img = arr['arr_0']
            rar_lime = arr['arr_1']
            adv_lime = arr['arr_2']
            gt = arr['arr_3']
            rar_lime=np.reshape(rar_lime,(8,8))
            adv_lime=np.reshape(adv_lime,(8,8))
            rar_lime=make_cam(rar_lime)
            adv_lime=make_cam(adv_lime)

            plt.figure()
            plt.subplot(1, 4, 1)
            plt.imshow(rar_lime)
            plt.subplot(1, 4, 2)
            plt.imshow(adv_lime)
            plt.subplot(1, 4, 3)
            plt.imshow(gt)
            plt.subplot(1, 4, 4)
            plt.imshow(img)
            plt.show()
            plt.close()
            if pred_label == adv_label:

                if get_rar_gt_iou(rar_lime, adv_lime, gt)==0:
                    continue

                defense_count += 1
                defense_rar_gt_iou += get_rar_gt_iou(rar_lime, adv_lime, gt)
                defense_iou += get_rar_adv_iou(rar_lime, adv_lime)

            else:

                attack_rar_gt_iou += get_rar_gt_iou(rar_lime, adv_lime, gt)
                attack_iou += get_rar_adv_iou(rar_lime, adv_lime)
                attack_count += 1
            total_num += 1
            print(pred_label, adv_label, get_rar_gt_iou(rar_lime, adv_lime, gt), get_rar_adv_iou(rar_lime, adv_lime))
            if i % 100 == 0:
                with open(results_file, 'a') as  f_w:
                    f_w.write(str(total_num) + " " + str(defense_count) + " " + str(attack_count) + " " + str(
                        attack_rar_gt_iou) + " " + str(attack_iou) + " " + str(defense_iou) + "\n")

    with open(results_file, 'a') as  f_w:
        f_w.write(str(total_num) + "defense_count " + str(defense_count) + "attack_count " + str(
            attack_count) + "attack_rar_gt_iou " + str(
            attack_rar_gt_iou) + "defense_rar_gt_iou " + str(
            defense_rar_gt_iou) + "attack_iou " + str(attack_iou) + "defense_iou " + str(
            defense_iou) + "\n")

        f_w.write(str(total_num) + " " + str(defense_count) + " " + str(attack_count) + " " + str(
            attack_rar_gt_iou / attack_count) + " " + str(
            defense_rar_gt_iou / defense_count) + " " + str(attack_iou / attack_count) + " " + str(
            defense_iou / defense_count) + "\n")

        print(str(total_num) + "defense_count " + str(defense_count) + "attack_count " + str(
            attack_count) + "attack_rar_gt_iou " + str(
            attack_rar_gt_iou) + "defense_rar_gt_iou " + str(
            defense_rar_gt_iou) + "attack_iou " + str(attack_iou) + "defense_iou " + str(
            defense_iou) + "\n")
        print(str(total_num) + " " + str(defense_count) + " " + str(attack_count) + " " + str(
            attack_rar_gt_iou / attack_count) + " " + str(
            defense_rar_gt_iou / defense_count) + " " + str(attack_iou / attack_count) + " " + str(
            defense_iou / defense_count) + "\n")
        # attack_rar_gt_iou<defense_rar_gt_iou
        # attack_iou<defense_iou

#
# 5449
# defense_count
# 935
# attack_count
# 4514
# attack_rar_gt_iou
# 1627.1751784620446
# defense_rar_gt_iou
# 324.71350454733243
# attack_iou
# 538.9858179518874
# defense_iou
# 177.3882843261956
# 7081
# defense_count
# 1025
# attack_count
# 6056
# attack_rar_gt_iou
# 2135.651181271406
# defense_rar_gt_iou
# 343.9010292269058
# attack_iou
# 839.2075584825325
# defense_iou
# 216.57206859439304
