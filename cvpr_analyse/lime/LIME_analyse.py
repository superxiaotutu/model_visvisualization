import os
import matplotlib.pyplot as plt
import numpy as np


def balance_rar_gt(rar, gt):
    new_rar = np.zeros((299, 299))
    rar_count = rar[rar == 2].size
    gt_count = gt[gt == 1].size

    area = (np.where(gt == 1))
    max_x = np.max(area[1])
    min_x = np.min(area[1])
    max_y = np.max(area[0])
    min_y = np.min(area[0])
    width = max_x - min_x
    height = max_y - min_y
    point_x = (max_x + min_x) // 2
    point_y = (max_y + min_y) // 2
    if rar_count < gt_count:
        scale = np.sqrt(rar_count / gt_count)
        width = width * scale / 2
        height = height * scale / 2
        new_x_1 = int(point_x - width) if int(point_x - width) > 0 else 0
        new_x_2 = int(point_x + width) if int(point_x + width) < 299 else 299
        new_y_1 = int(point_y - height) if int(point_y - height) > 0 else 0
        new_y_2 = int(point_y + height) if int(point_y + height) < 299 else 299
        new_rar[new_y_1:new_y_2, new_x_1:new_x_2, ] = 1
    else:
        scale = np.sqrt(rar_count / gt_count)
        width = width * scale / 2
        height = height * scale / 2
        new_x_1 = int(point_x - width) if int(point_x - width) > 0 else 0
        new_x_2 = int(point_x + width) if int(point_x + width) < 299 else 299
        new_y_1 = int(point_y - height) if int(point_y - height) > 0 else 0
        new_y_2 = int(point_y + height) if int(point_y + height) < 299 else 299
        new_rar[new_y_1:new_y_2,
        new_x_1:new_x_2, ] = 1
    return new_rar


def get_rar_gt_iou(rar, adv, ground_truth):
    ground_truth = balance_rar_gt(rar, ground_truth)
    ground_count = ground_truth[ground_truth == 1].size
    rar_sum = rar + ground_truth
    # adv_sum = adv + ground_truth
    rar_IOU = rar_sum[rar_sum == 3].size / ground_count
    return rar_IOU


def get_rar_adv_iou(rar, adv):
    sum = rar + adv
    IOU = sum[sum == 4].size / sum[sum != 0].size
    return IOU


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

                filename = f.split('_')
                pred_label = filename[0]
                adv_label = filename[1]
                img = arr['arr_0']
                rar_lime = arr['arr_1']
                adv_lime = arr['arr_2']
                gt = arr['arr_3']
                rar_lime[rar_lime == 1] = 0
                adv_lime[adv_lime == 1] = 0
                rar_lime[rar_lime == 3] = 0
                adv_lime[adv_lime == 3] = 0
                if pred_label == adv_label:
                    if get_rar_gt_iou(rar_lime, adv_lime, gt) == 0:
                        continue

                    defense_count += 1
                    defense_rar_gt_iou += get_rar_gt_iou(rar_lime, adv_lime, gt)
                    defense_iou += get_rar_adv_iou(rar_lime, adv_lime)

                else:
                    if get_rar_gt_iou(rar_lime, adv_lime, gt) == 1:
                        continue
                    attack_rar_gt_iou += get_rar_gt_iou(rar_lime, adv_lime, gt)
                    attack_iou += get_rar_adv_iou(rar_lime, adv_lime)
                    attack_count += 1
                # plt.figure()
                # plt.subplot(1, 4, 1)
                # plt.imshow(rar_lime)
                # plt.subplot(1, 4, 2)
                # plt.imshow(adv_lime)
                # plt.subplot(1, 4, 3)
                # plt.imshow(gt)
                # plt.subplot(1, 4, 4)
                # plt.imshow(balance_rar_gt(rar_lime, gt))
                # plt.show()
                total_num += 1
                print(pred_label, adv_label, get_rar_gt_iou(rar_lime, adv_lime, gt), get_rar_adv_iou(rar_lime, adv_lime))
                if i % 100 == 0:
                    with open(results_file, 'a') as  f_w:
                        f_w.write(str(total_num) + " " + str(defense_count) + " " + str(attack_count) + " " + str(
                            attack_rar_gt_iou) + " " + str(attack_iou) + " " + str(defense_iou) + "\n")
            except Exception as e:
                print(e)
                print(f)
                continue
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
