# # import numpy
import tensorflow as tf
# # a=tf.placeholder(tf.float32,[8,8])
# # b=tf.reshape(a,[64])
# # c=tf.nn.top_k(b, k=10).values[-1]
# #
# # sess=tf.Session()
# # arr=numpy.asarray([[3.8073652e-33, 4.3125332e-33, 3.6105485e-33, 4.0398604e-33, 3.2486711e-33,
# #   3.4646363e-33, 2.2971766e-33, 2.2527280e-33,],
# #  [2.1365532e-33, 3.3336116e-33, 1.0682583e-32, 3.7084934e-29, 6.4680172e-29,
# #   1.9050530e-31, 6.5972850e-33, 4.4185422e-33,],
# #  [3.6774371e-33, 7.4327971e-32, 5.2131503e-26, 2.4591440e-14 ,4.4862725e-12,
# #   9.2481092e-18, 4.0759869e-27, 5.4097296e-32,],
# #  [4.2867821e-33, 1.7385526e-30 ,1.6248188e-21, 2.5487682e-07, 9.9999821e-01,
# #   1.5055002e-06 ,3.5418707e-21 ,6.8587895e-31,],
# #  [5.9149414e-33, 2.6919631e-30, 2.0173552e-24, 1.5537206e-16, 1.9134035e-12,
# #   4.7926839e-16 ,1.8539056e-26, 2.5126506e-32,],
# #  [5.3458655e-33, 3.3268424e-32, 4.2535750e-30, 9.8844695e-29, 4.8832305e-28,
# #   3.2262751e-29, 1.5793617e-32, 3.5940589e-33,],
# #  [3.7396255e-33, 4.3051037e-33, 7.0722942e-33, 6.0425192e-33, 4.9835028e-33,
# #   5.1045251e-33, 4.4853856e-33, 2.4628003e-33,],
# #  [4.1031350e-33, 3.2865393e-33, 3.7592493e-33, 4.3505369e-33, 3.9368349e-33,
# #   3.1352242e-33, 3.3703348e-33, 1.9859942e-33,]])
# # a=sess.run(c,feed_dict={a:arr})
# # print(a)
#
# import os
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def get_rar_gt_iou(rar,adv, ground_truth):
#     ground_count=ground_truth[ground_truth==1].size
#     rar_sum = rar + ground_truth
#     # adv_sum = adv + ground_truth
#     rar_IOU = rar_sum[rar_sum == 3].size / ground_count
#     return rar_IOU
#
#
# def get_rar_adv_iou(rar, adv):
#     sum = rar + adv
#     IOU = sum[sum == 4].size / sum[sum != 0].size
#     return IOU
#
#
# if __name__ == '__main__':
#     results_file = 'lime_iou.txt'
#     temp_rar = 0
#     temp_adv = 0
#     defense_rar_gt_iou = 0
#     defense_adv_gt_iou = 0
#     attack_adv_gt_iou = 0
#     attack_rar_gt_iou = 0
#     defense_iou = 0
#     attack_iou = 0
#     defense_count = 0
#     attack_count = 0
#     total_num = 0
#     if os.path.isfile(results_file):
#         os.remove(results_file)
#     for root, dirs, files in os.walk('npz'):
#
#         for i, f in enumerate(files):
#             try:
#                 arr = np.load("npz/" + f)
#             except Exception as e:
#                 print(e)
#                 continue
#             filename = f.split('_')
#             pred_label = filename[0]
#             adv_label = filename[1]
#             img = arr['arr_0']
#             rar_lime = arr['arr_1']
#             adv_lime = arr['arr_2']
#             gt = arr['arr_3']
#
#             rar_lime[rar_lime == 1] = 0
#             adv_lime[adv_lime == 1] = 0
#             rar_lime[rar_lime == 3] = 0
#             adv_lime[adv_lime == 3] = 0
#             print(rar_lime)
#             rar_lime=np.reshape(rar_lime,(8,8))
#             adv_lime=np.reshape(adv_lime,(8,8))
#
#             plt.figure()
#             plt.subplot(1, 4, 1)
#             plt.imshow(img)
#             plt.subplot(1, 4, 2)
#             plt.imshow(adv_lime)
#             plt.subplot(1, 4, 3)
#             plt.imshow(gt)
#             plt.subplot(1, 4, 4)
#             plt.imshow(rar_lime)
#             plt.show()
#             plt.close()
sess=tf.Session()

mat=[[[[1],[2]],[[4],[3]]],[[[1],[2]],[[2],[5]]],[[[1],[2]],[[2],[10]]]]
mat=tf.reshape(mat,[3,2,2])
#
a=tf.reshape(mat,[3,4])
a=tf.nn.top_k(a,k=3)
t=a.values[:,-1]
t = tf.expand_dims(t, 1)
t = tf.expand_dims(t, 1)
t=tf.tile(t,[1,2,2])

r=mat-t
print(sess.run(mat))
print(sess.run(t))
print(sess.run(r))