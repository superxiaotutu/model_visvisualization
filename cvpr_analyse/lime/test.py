import cv2
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf

# sess=tf.Session()
#
# b=[[[0.1075917],
#   [0.01612537],
#   [0.01716102],
#   [0.01706656],
#   [0.01712777],
#   [0.01714061],
#   [0.01653875],
#   [0.01586068],],
# ]
# a=[[[0.21587086],
#   [0.11542621],
#   [0.01634356],
#   [0.01637067],
#   [0.01623338],
#   [0.01611543],
#   [0.01543998],
#   [0.01559884],],
#
# ],
# a = a - tf.reduce_mean(a)
# a = tf.nn.relu(a)
# a = tf.sign(a)
# b = b - tf.reduce_mean(b)
# b = tf.nn.relu(b)
# b = tf.sign(b)
# sum = a+b
# count_no_0 = tf.count_nonzero(sum)
# count_no_0 = tf.cast(count_no_0, tf.float32)
# temp_sub_2 = tf.subtract(sum, tf.cast(2,tf.float32))
# temp_sub_2_nonzero = tf.count_nonzero(temp_sub_2)
# count_2 = tf.subtract(tf.cast(8,tf.int64), temp_sub_2_nonzero)
# count_2 = tf.cast(count_2, tf.float32)
# IOU = tf.divide(count_2, count_no_0)
# print(sess.run(count_2))
# print(sess.run(count_no_0))
# print(sess.run(IOU))
#
# print(sess.run(sum))
from skimage.segmentation import mark_boundaries

np.set_printoptions(threshold=100000)
img=np.load('1.npz')
print(img)
plt.figure()
plt.subplot(1,4,1)
plt.imshow(img['arr_0'])
plt.subplot(1,4,2)
plt.imshow(img['arr_1'])
plt.subplot(1,4,3)
plt.imshow(img['arr_2'])
plt.subplot(1,4,4)
plt.imshow(img['arr_3'])
plt.show()
# a=np.load('adv.npy')
# a = cv2.applyColorMap(np.uint8(255 * a), cv2.COLORMAP_JET)
# a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
# plt.imshow(a)
# plt.show()
# img = img.astype(float)
# img /= img.max()
# alpha = 2
# rar = img + alpha * a
# plt.imshow((rar))
# plt.show()
# print(a)
