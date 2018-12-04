import tensorflow as tf
import  numpy as np
import  matplotlib.pyplot as plt
# a=np.load('a.npy')
# b=np.load('b.npy')
# np.set_printoptions(threshold=10000)
# print(a==b)
# print(a[0].shape)
# plt.imshow(a[0])
# plt.show()
# plt.imshow(b[0])
#
# plt.show()
# 默认会话
sess = tf.Session()
# constant_2=tf.constant(2,dtype=tf.float32)
# constant_64=tf.constant(8,dtype=tf.int64)
# constant_03=tf.constant(0.2,dtype=tf.float32)
#
# # 定义两个list
rar = [[[[0],[1]],[[0],[1]]],[[[1],[0]],[[0],[1]]]]
adv = [[[[1],[1]],[[1],[1]]],[[[1],[1]],[[0],[0]]]]
rar=tf.cast(rar,tf.float32)
adv=tf.cast(adv,tf.float32)

sum = tf.add(rar, adv)

constant_2 = tf.constant(2, dtype=tf.float32)
constant_64 = tf.constant(4, dtype=tf.int64)

count_no_0 = tf.count_nonzero(sum, [1, 2, 3])
count_no_0 = tf.cast(count_no_0, tf.float32)
temp_sub_2 = tf.subtract(sum, constant_2)
temp_sub_2_nonzero = tf.count_nonzero(temp_sub_2, [1, 2, 3])
count_2 = tf.subtract(constant_64, temp_sub_2_nonzero)
count_2 = tf.cast(count_2, tf.float32)
IOU = tf.divide(count_2, count_no_0)
IOU = IOU-0.1
# a=tf.cast(a,dtype=tf.float32)
# b=tf.cast(b,dtype=tf.float32)
print(sess.run(count_2))

print(sess.run(IOU))
print(sess.run(sum))
# # is_iou_exceeds = tf.greater(constant_2,constant_03)
# # is_iou_exceeds = tf.cast(is_iou_exceeds, dtype=tf.float32)
#
# # print(a)
# sum = tf.add(a, b)
# count_no_0 = tf.count_nonzero(sum[:,:,:])
# temp_sub_2 = tf.subtract(sum, constant_2)
# temp_sub_2_nonzero = tf.count_nonzero(temp_sub_2)
# count_2 = tf.subtract(constant_64, temp_sub_2_nonzero)
# count_2 = tf.cast(count_2, tf.float32)
# count_no_0 = tf.cast(count_no_0, tf.float32)
# IOU = tf.divide(count_2, count_no_0)
# IOU=count_2 /count_no_0
# print(sess.run(sum))
# print(sess.run(count_no_0))
# print(sess.run(IOU))
#
# print(sess.run(IOU))

# greater(x, y, name=None)
# x: 一个 tensor或是一个list等。数据必须是以下的数据类型: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
# y: 一个 tensor或是一个list等，需要和x保持同样的数据类型。
# c = tf.greater(a,b)

# 输出
