import os
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cifarnet
import numpy as np
from skimage.transform import resize
import tensorflow.contrib.slim.nets as nets

import sys

attack_step=0.05
a, b, c, cuda = 8, 4, 1, '1'
key=10
print(a, b, c, cuda)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda
a, b, c, d = float(a), float(b), float(c), 1

_BATCH_SIZE = 50
X = tf.placeholder(tf.float32, [_BATCH_SIZE, 299, 299, 3])
X_adv = tf.placeholder(tf.float32, [_BATCH_SIZE, 299, 299, 3])
IOUS = tf.placeholder(tf.float32, [_BATCH_SIZE])

Y = tf.placeholder(tf.int32, [_BATCH_SIZE])
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)


def inception(image, reuse=tf.AUTO_REUSE):
    preprocessed = tf.multiply(tf.subtract(image, 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, end_point = nets.inception.inception_v3(preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


def step_target_class_adversarial_images(x, eps, one_hot_target_class):
    logits, probs, end_points = inception(x)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                    logits,
                                                    label_smoothing=0.1,
                                                    weights=1.0)
    IOUS_ = tf.expand_dims(IOUS, 1)
    IOUS_ = tf.expand_dims(IOUS_, 1)
    IOUS_ = tf.expand_dims(IOUS_, 1)
    IOUS_ = tf.tile(IOUS_, [1, 299, 299, 3])
    x_adv = x - eps * tf.sign(tf.gradients(cross_entropy, x)[0]) * IOUS_
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
    return tf.stop_gradient(x_adv)


def stepll_adversarial_images(x, eps):
    logits, prob, end_points = inception(x)
    least_likely_class = tf.argmin(logits, 1)
    one_hot_ll_class = tf.one_hot(least_likely_class, 1000)
    return step_target_class_adversarial_images(x, eps, one_hot_ll_class)

def grad_cam(end_point, pre_calss_one_hot, layer_name='Mixed_7c'):
    pre_calss_one_hot = tf.one_hot(Y, depth=1000)
    conv_layer = end_point[layer_name]
    signal = tf.multiply(end_point['Logits'][:, 1:], pre_calss_one_hot)
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

    resize_cam = tf.image.resize_images(cam, [299, 299])
    resize_cam = resize_cam / tf.reduce_max(resize_cam)

    cam = cam - tf.reduce_mean(cam)
    cam = tf.nn.relu(cam)
    cam = tf.sign(cam)
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
    saver.restore(sess, "model" + str(a) + '_' + str(b) + '_' + str(c) + "/model.ckpt")
    # saver.restore(sess, "../calculate_iou/inceptionv3.ckpt")


constant_2 = tf.constant(2, dtype=tf.float32)
constant_64 = tf.constant(64, dtype=tf.int64)


def get_rar_adv_iou(rar, adv):
    sum = tf.add(rar, adv)
    count_no_0 = tf.count_nonzero(sum, [1, 2, 3])
    count_no_0 = tf.cast(count_no_0, tf.float32)
    temp_sub_2 = tf.subtract(sum, constant_2)
    temp_sub_2_nonzero = tf.count_nonzero(temp_sub_2, [1, 2, 3])
    count_2 = tf.subtract(constant_64, temp_sub_2_nonzero)
    count_2 = tf.cast(count_2, tf.float32)
    IOU = tf.divide(count_2, count_no_0)
    IOU = IOU - key
    IOU = tf.nn.relu(IOU)
    IOU = tf.sign(IOU)
    return IOU



# 主要分类节点
fixed_adv_sample_get_op = stepll_adversarial_images(X, attack_step)

rar_logits, rar_probs, rar_end_point = inception(X)
adv_logits, adv_probs, adv_end_point = inception(X_adv)
# gradcam并判断iou

rar_grad_cam, rar_cam = grad_cam(rar_end_point, Y)
adv_grad_cam, adv_cam = grad_cam(adv_end_point, Y)
rar_adv_ious = get_rar_adv_iou(rar_cam, adv_cam)

correct_p = tf.equal(tf.argmax(rar_probs, 1), (tf.argmax(Y, 1)))
accuracy = tf.reduce_mean(tf.cast(correct_p, "float"))


sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

saver.restore(sess, "../modelinception_v3.ckpt")

sess.graph.finalize()
time_start = time.time()
import PIL
def load_img(path):
    I = PIL.Image.open(path).convert('RGB')
    I = I.resize((299, 299)).crop((0, 0, 299, 299))
    I = (np.asarray(I) / 255.0).astype(np.float32)
    return I[:, :, 0:3]

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
result_file = 'find_iou_imagenet_'+str(key)+ '_'+str(attack_step)+'.txt'

if os.path.isfile(result_file):
    os.remove(result_file)
sum_1 = 0
sum_2 = 0
sum_3 = 0
loop_num = 0
IOU_sum = 0
labels_file = 'imagenet_labels.txt'

if os.path.exists(result_file):
    os.remove(result_file)
defense_iou = 0
defense_count = 0
attack_iou = 0
attack_count = 0
rar_ground_iou_sum = 0
adv_ground_iou_sum = 0

label_paths = []
train_batch=[]
with open(labels_file, 'r', encoding='utf-8')as f:
    lines = f.readlines()
    for index, line in enumerate(lines):
        imgs = []
        labels = []
        label_letter = line.split(' ')
        ground_truths = []
        label_letter = label_letter[0]
        img_class = index
        dir_name = 'img_val/' + str(label_letter)
        for root, dirs, files in os.walk(dir_name):
            for file in files:
                img_path = dir_name + '/' + file
                label_path = 'val/' + str(file)[:-4] + 'xml'
                imgs.append(load_img(img_path))
                labels.append(index)

    train_batch[0]=imgs
    train_batch[1]=labels
for i in range(50000 // _BATCH_SIZE):
    batch = sess.run(train_batch)
    adv_sample = sess.run(fixed_adv_sample_get_op, feed_dict={X: batch[0]})
    ious = sess.run(rar_adv_ious, feed_dict={X: batch[0], X_adv: adv_sample, Y: batch[1]})
    acc = sess.run(accuracy,feed_dict={X: batch[0], Y: batch[1]})
    with open(result_file, 'a') as f:
        f.write('1' + " " + str(ious) + "\n")
    sum_1 += ious
    adv_sample_N1 = sess.run(fixed_adv_sample_get_op, feed_dict={X: adv_sample})
    ious = sess.run(rar_adv_ious, feed_dict={X: batch[0], X_adv: adv_sample_N1, Y: batch[1]})
    with open(result_file, 'a') as f:
        f.write('2' + " " + str(ious) + "\n")
    sum_2 += ious

    adv_sample_N2 = sess.run(fixed_adv_sample_get_op, feed_dict={X: adv_sample_N1})
    ious = sess.run(rar_adv_ious, feed_dict={X: batch[0], X_adv: adv_sample_N2, Y: batch[1]})
    with open(result_file, 'a') as f:
        f.write('3' + " " + str(ious) + "\n")
    sum_3 += ious
    print(i,sum_1,sum_2,sum_3)
print(sum_1 / 50000, sum_2 / 50000, sum_3 / 50000, )

with open(result_file, 'a') as f:
    f.write('1_' + " " + str(sum_1 / 50000) + "\n")
    f.write('2' + " " + str(sum_2 / 50000) + "\n")
    f.write('3' + " " + str(sum_3 / 50000) + "\n")
