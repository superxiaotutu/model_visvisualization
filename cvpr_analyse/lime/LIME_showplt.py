import os
import matplotlib.pyplot as plt
import numpy as np
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
            arr = np.load("npz/" + f)
            filename = f.split('_')
            pred_label = filename[0]
            adv_label = filename[1]
            img = arr['arr_0']
            print(img)
            rar_lime = arr['arr_1']
            adv_lime = arr['arr_2']
            # gt = arr['arr_3']
            rar_lime[rar_lime == 1] = 0
            adv_lime[adv_lime == 1] = 0
            rar_lime[rar_lime == 3] = 0
            adv_lime[adv_lime == 3] = 0
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(img+np.max(img)/np.max(img))
            plt.subplot(1, 3, 2)
            plt.imshow(rar_lime)
            plt.subplot(1, 3, 3)
            plt.imshow(adv_lime)
            plt.show()
