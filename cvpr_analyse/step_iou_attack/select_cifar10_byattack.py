import os
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cifarnet
import numpy as np
from skimage.transform import resize

import sys

a, b, c, cuda = 8, 4, 1.6, '-1'

print(a, b, c, cuda)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda

a, b, c, d = float(a), float(b), float(c), 1

_BATCH_SIZE = 100

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
X_adv = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 10])


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






def save():
    saver = tf.train.Saver()
    saver.save(sess, "model" + str(a) + '_' + str(b) + '_' + str(c) + "/model.ckpt")


def restore():
    saver = tf.train.Saver()
    saver.restore(sess, "model" + '8.0' + '_' + '4.0' + '_' + '1.6' + "/model.ckpt")
    # saver.restore(sess, "../calculate_iou/inceptionv3.ckpt")



# 主要分类节点
fixed_adv_sample_get_op = stepll_adversarial_images(X, 0.15)
rar_logits, rar_probs, rar_end_point = bulid_Net(X)
adv_logits, adv_probs, adv_end_point = bulid_Net(X_adv)
# gradcam并判断iou
is_correct = tf.equal(tf.argmax(rar_probs, 1), (tf.argmax(Y, 1)))
is_defense = tf.equal(tf.argmax(rar_probs, 1), (tf.argmax(adv_probs, 1)))




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

    img_batch = tf.train.batch([image, Label], batch_size=batch_size, num_threads=64
                               )
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
adv_writer = tf.summary.FileWriter("model/log/adv", sess.graph)


restore()

key = 0.59513

time_start = time.time()
result_file = 'result' + str(a) + '_' + str(b) + '_' + str(c) + "-" + str(key) + '.txt'




import matplotlib.pyplot as plt


sum_1 = 0
sum_2 = 0
sum_3 = 0
attack_3_count = 0
attack_2_count = 0
total_num = 0
iou_attack_count = 0
iou_loop_num = 0
tfrecords_filename = '0.15_selected_sample.tfrecords'
if os.path.isfile(tfrecords_filename):
    os.remove(tfrecords_filename)
writer = tf.python_io.TFRecordWriter(tfrecords_filename)
for i in range(50000 // _BATCH_SIZE):
    batch = sess.run(train_batch)
    is_corrects = sess.run(is_correct, feed_dict={X: batch[0], Y: batch[1]})
    for j in range(_BATCH_SIZE):
        if is_corrects[j]:
            sum_1 += 1
            adv_sample = sess.run(fixed_adv_sample_get_op, feed_dict={X: [batch[0][j]]})
            is_defenses = sess.run(is_defense, feed_dict={X: [batch[0][j]], X_adv: adv_sample, Y: batch[1]})
            if not is_defenses[0]:
                img_raw = batch[0][j]
                img_label = batch[1][j]
                img_label_number = np.argmax(img_label)
                img_label_number = int(img_label_number)
                img_raw = img_raw.tostring()
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_label_number])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }))
                writer.write(example.SerializeToString())
                sum_2 += 1
    print(sum_1,sum_2)
