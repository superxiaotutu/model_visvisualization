import os
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cifarnet
import numpy as np
from skimage.transform import resize

import sys



a, b, c, cuda = 8, 4, 1.6, '1'

print(a, b, c, cuda)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda

a, b, c, d = float(a), float(b), float(c), 1

_BATCH_SIZE = 1

X = tf.placeholder(tf.float32, [_BATCH_SIZE, 32, 32, 3])
X_adv = tf.placeholder(tf.float32, [_BATCH_SIZE, 32, 32, 3])
Y = tf.placeholder(tf.float32, [_BATCH_SIZE, 10])


def bulid_Net(image):
    image = tf.reshape(image, [-1, 32, 32, 3])
    with tf.variable_scope(name_or_scope='CifarNet', reuse=tf.AUTO_REUSE):
        arg_scope = cifarnet.cifarnet_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_point = cifarnet.cifarnet(image)
            probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


def step_target_class_adversarial_images(x, eps, one_hot_target_class):
    logits, probs, end_points = bulid_Net(x)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                    logits,
                                                    label_smoothing=0.1,
                                                    weights=1.0)
    x_adv = x - eps * tf.sign(tf.gradients(cross_entropy, x)[0])
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
    return tf.stop_gradient(x_adv)


def stepll_adversarial_images(x, eps):
    logits, probs, end_points = bulid_Net(x)
    least_likely_class = tf.argmin(logits, 1)
    one_hot_ll_class = tf.one_hot(least_likely_class, 10)
    return step_target_class_adversarial_images(x, eps, one_hot_ll_class)


def stepllnoise_adversarial_images(x, eps):
    logits, probs, end_points = bulid_Net(x)
    least_likely_class = tf.argmin(logits, 1)
    one_hot_ll_class = tf.one_hot(least_likely_class, 10)
    x_noise = x + eps / 2 * tf.sign(tf.random_normal(x.shape))
    return step_target_class_adversarial_images(x_noise, eps / 2, one_hot_ll_class)


def grad_cam(end_point, pre_calss_one_hot, layer_name='pool2'):
    conv_layer = end_point[layer_name]
    signal = tf.multiply(end_point['Logits'], pre_calss_one_hot)
    loss = tf.reduce_mean(signal, 1)
    grads = tf.gradients(loss, conv_layer)[0]
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))
    weights = tf.reduce_mean(norm_grads, axis=(1, 2))
    weights = tf.expand_dims(weights, 1)
    weights = tf.expand_dims(weights, 1)
    weights = tf.tile(weights, [1, 8, 8, 1])
    pre_cam = tf.multiply(weights, conv_layer)
    cam = tf.reduce_sum(pre_cam, 3)
    cam = tf.expand_dims(cam, 3)
    """"""
    cam = tf.reshape(cam, [-1, 64])
    cam = tf.nn.softmax(cam)
    cam = tf.reshape(cam, [-1, 8, 8, 1])

    resize_cam = tf.image.resize_images(cam, [32, 32])
    resize_cam = resize_cam / tf.reduce_max(resize_cam)

    cam = cam - tf.reduce_mean(cam)
    cam = tf.nn.relu(cam)
    cam = tf.sign(cam)

    # resize_cam = tf.image.resize_images(cam, [32, 32])
    # resize_cam = tf.sign(resize_cam)
    # 放大后原始是1的会变成0。5，所以要除以0.5，这样又回去了，保持数据不变
    # resize_cam = resize_cam / tf.reduce_max(resize_cam)

    return resize_cam, cam


def mask_image(img, grad_cam):
    img = tf.reshape(img, [-1, 32, 32, 3])
    cam_clip = tf.sign(tf.nn.relu(grad_cam - tf.reduce_mean(grad_cam)))
    reverse_cam = tf.abs(cam_clip - 1)
    return reverse_cam * img


def save():
    saver = tf.train.Saver()
    saver.save(sess, "model" + str(a) + '_' + str(b) + '_' + str(c) + "/model.ckpt")


def restore():
    saver = tf.train.Saver()
    saver.restore(sess, "model" + '8.0' + '_' + '4.0' + '_' + '1.6' + "/model.ckpt")
    # saver.restore(sess, "../calculate_iou/inceptionv3.ckpt")


constant_2 = tf.constant(2, dtype=tf.float32)
constant_64 = tf.constant(64, dtype=tf.int64)


def get_rar_adv_iou(rar, adv):
    sum = tf.add(rar, adv)
    count_no_0 = tf.count_nonzero(sum)
    count_no_0 = tf.cast(count_no_0, tf.float32)
    temp_sub_2 = tf.subtract(sum, constant_2)
    temp_sub_2_nonzero = tf.count_nonzero(temp_sub_2)
    count_2 = tf.subtract(constant_64, temp_sub_2_nonzero)
    count_2 = tf.cast(count_2, tf.float32)
    IOU = tf.divide(count_2, count_no_0)
    return IOU


# 主要分类节点
fixed_adv_sample_get_op = stepll_adversarial_images(X, 0.05)
rar_logits, rar_probs, rar_end_point = bulid_Net(X)
adv_logits, adv_probs, adv_end_point = bulid_Net(X_adv)
# gradcam并判断iou
is_correct = tf.equal(tf.argmax(rar_probs, 1), (tf.argmax(Y, 1)))
is_defense = tf.equal(tf.argmax(rar_probs, 1), (tf.argmax(adv_probs, 1)))
rar_grad_cam, rar_cam = grad_cam(rar_end_point, Y)
adv_grad_cam, adv_cam = grad_cam(adv_end_point, Y)
constant_03 = tf.constant(0.3, dtype=tf.float32)
rar_adv_ious = get_rar_adv_iou(rar_cam, adv_cam)


# is_iou_exceeds = tf.greater(rar_adv_iou, constant_03)
# is_iou_exceeds = tf.cast(is_iou_exceeds, dtype=tf.float32)


# 进攻
# adv_sample_get_op = stepll_adversarial_images(X, tf.random_uniform([1], 0, 0.3))
# fixed_adv_sample_get_op = stepll_adversarial_images(X, 0.15)
# NOISE_adv_sample_get_op = stepllnoise_adversarial_images(X, 0.15)


def read_and_decode(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    feature = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64))}

    features = tf.parse_single_example(serialized_example, features=feature)

    Label = tf.one_hot(features['image/class/label'], 10)
    A = tf.image.decode_png(features['image/encoded'])

    image = tf.reshape(A, shape=[32, 32, 3])

    image = tf.image.random_flip_left_right(image)

    preprocessed = tf.multiply(tf.subtract(image / 255, 0.5), 2.0)

    img_batch = tf.train.shuffle_batch([preprocessed, Label], batch_size=batch_size, num_threads=64, capacity=10000,
                                       min_after_dequeue=5000)
    return img_batch


def test_read_and_decode(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    feature = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64))}

    features = tf.parse_single_example(serialized_example, features=feature)

    Label = tf.one_hot(features['image/class/label'], 10)
    A = tf.image.decode_png(features['image/encoded'])

    A = tf.reshape(A, shape=[32, 32, 3])

    preprocessed = tf.multiply(tf.subtract(A / 255, 0.5), 2.0)

    img_batch = tf.train.batch([preprocessed, Label], batch_size=batch_size, num_threads=64, capacity=1000)
    return img_batch


def make_cam(x):
    x = np.reshape(x, (8, 8))
    threshold = np.sort(np.reshape(x, (64)))[-int(key)]
    x = x - threshold
    x = resize(x, [32, 32])
    x[x < 0] = 0
    x[x > 0] = 1
    return x


filename = 'cifar_data/cifar10_train.tfrecord'
test_filename = 'cifar_data/cifar10_test.tfrecord'
train_batch = read_and_decode(filename, _BATCH_SIZE)
test_batch = test_read_and_decode(test_filename, _BATCH_SIZE)
coord = tf.train.Coordinator()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

threads = tf.train.start_queue_runners(coord=coord, sess=sess)

train_writer = tf.summary.FileWriter("model/log/train", sess.graph)
# test_writer = tf.summary.FileWriter("model/log/test", sess.graph)
adv_writer = tf.summary.FileWriter("model/log/adv", sess.graph)
#
# try:
restore()
# except:
#     print('init')
#     pass
key=0.59513

time_start = time.time()
result_file = 'result' + str(a) + '_' + str(b) + '_' + str(c) +"-"+str(key)+ '.txt'

f_w = open(result_file, 'a')
f_w.write('ini')
f_w.close()


def show_img(rar, adv):
    print(rar_adv_ious)
    plt.figure()
    plt.subplot(1, 5, 1)
    plt.imshow(batch[0][0])
    plt.subplot(1, 5, 2)
    # adv = np.reshape(advs[0], (8, 8))
    # plt.imshow(adv)
    # plt.subplot(1, 5, 3)
    # rar = np.reshape(rars[0], (8, 8))
    # plt.imshow(rar)
    #
    # plt.subplot(1, 5, 4)
    # rar1 = np.reshape(rars[0], (8, 8))
    # plt.imshow(make_cam(rar1))
    # plt.subplot(1, 5, 5)
    # adv1 = np.reshape(advs[0], (8, 8))
    # plt.imshow(make_cam(adv1))
    plt.show()


import matplotlib.pyplot as plt

if os.path.isfile(result_file):
    os.remove(result_file)
sum_1 = 0
sum_2 = 0
sum_3 = 0
attack_3_count = 0
attack_2_count = 0
total_num = 0
iou_attack_count = 0
iou_loop_num = 0
for i in range(50000 // _BATCH_SIZE):
    batch = sess.run(train_batch)
    step = 0
    adv_sample = sess.run(fixed_adv_sample_get_op, feed_dict={X: batch[0]})
    ious = sess.run(rar_adv_ious, feed_dict={X: batch[0], X_adv: adv_sample, Y: batch[1]})
    while True:
        if ious > key or step == 4:
            adv_sample = sess.run(fixed_adv_sample_get_op, feed_dict={X: adv_sample})
            ious = sess.run(rar_adv_ious, feed_dict={X: batch[0], X_adv: adv_sample, Y: batch[1]})
            step += 1
        else:
            break
    is_defenses = sess.run(is_defense, feed_dict={X: batch[0], X_adv: adv_sample, Y: batch[1]})
    if not is_defenses[0]:
        iou_attack_count += 1
    iou_loop_num += 1

    adv_sample = sess.run(fixed_adv_sample_get_op, feed_dict={X: batch[0]})
    adv_sample = sess.run(fixed_adv_sample_get_op, feed_dict={X: adv_sample})
    adv_sample = sess.run(fixed_adv_sample_get_op, feed_dict={X: adv_sample})
    is_defenses = sess.run(is_defense, feed_dict={X: batch[0], X_adv: adv_sample, Y: batch[1]})
    if not is_defenses[0]:
        attack_3_count += 1
    adv_sample = sess.run(fixed_adv_sample_get_op, feed_dict={X: batch[0]})
    adv_sample = sess.run(fixed_adv_sample_get_op, feed_dict={X: adv_sample})
    is_defenses = sess.run(is_defense, feed_dict={X: batch[0], X_adv: adv_sample, Y: batch[1]})
    if not is_defenses[0]:
        attack_2_count += 1
    total_num += 1

    with open(result_file, 'a') as f:
        f.write(str(i)+" "+str(iou_attack_count) + " " + str(iou_loop_num) + " " + str(attack_2_count) + " " + str(
            attack_3_count) + " " + str(total_num) + " " + "\n")
    print(i,iou_attack_count, attack_2_count,attack_3_count)

# with open(result_file, 'a') as f:
#     f.write('1' + " " + str(iou_attack_count / iou_loop_num) + "\n")
#     f.write('2' + " " + str(attack_2_count / total_num) + "\n")
#     f.write('3' + " " + str(attack_3_count / total_num) + "\n")
# print(ious)
#     with open(result_file, 'a') as f:
#         f.write('1' + " " + str(ious) + "\n")
#     sum_1 += ious
#     adv_sample_N1 = sess.run(fixed_adv_sample_get_op, feed_dict={X: adv_sample})
#     ious = sess.run(rar_adv_ious, feed_dict={X: batch[0], X_adv: adv_sample_N1, Y: batch[1]})
#     with open(result_file, 'a') as f:
#         f.write('2' + " " + str(ious) + "\n")
#     sum_2 += ious
#
#     adv_sample_N2 = sess.run(fixed_adv_sample_get_op, feed_dict={X: adv_sample_N1})
#     ious = sess.run(rar_adv_ious, feed_dict={X: batch[0], X_adv: adv_sample_N2, Y: batch[1]})
#     with open(result_file, 'a') as f:
#         f.write('3' + " " + str(ious) + "\n")
#     sum_3 += ious
# print(sum_1 / 50000, sum_2 / 50000, sum_3 / 50000, )
#
# with open(result_file, 'a') as f:
#     f.write('1' + " " + str(sum_1 / 50000) + "\n")
#     f.write('2' + " " + str(sum_2 / 50000) + "\n")
#     f.write('3' + " " + str(sum_3 / 50000) + "\n")
