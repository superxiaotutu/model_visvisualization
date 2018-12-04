import os
import random

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from skimage.transform import resize

import PIL
import numpy as np
import json
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

plt.switch_backend('agg')

sess = tf.InteractiveSession()
image = tf.Variable(tf.zeros((299, 299, 3)))


# 加载inceptionV
def inception(image, reuse):
    preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, end_point = nets.inception.inception_v3(preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


logits, probs, end_point = inception(image, reuse=False)

restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV3/')
]
saver = tf.train.Saver(restore_vars)
saver.restore(sess, "inception_v3.ckpt")

imagenet_json = 'imagenet.json'
with open(imagenet_json) as f:
    imagenet_labels = json.load(f)


# 打印进攻前的图片
def classify(img, correct_class=None, target_class=None):
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    # fig.sca(ax1)
    p = sess.run(probs, feed_dict={image: img})[0]
    # ax1.imshow(img)
    # fig.sca(ax1)
    #
    # topk = list(p.argsort()[-10:][::-1])
    # topprobs = p[topk]
    # barlist = ax2.bar(range(10), topprobs)
    # if target_class in topk:
    #     barlist[topk.index(target_class)].set_color('r')
    # if correct_class in topk:
    #     barlist[topk.index(correct_class)].set_color('g')
    # plt.sca(ax2)
    # plt.ylim([0, 1.1])
    # plt.xticks(range(10),
    #            [imagenet_labels[i][:15] for i in topk],
    #            rotation='vertical')
    # fig.subplots_adjust(bottom=0.2)
    # plt.close()
    return np.argmax(p)


# 进攻
def step_target_class_adversarial_images(x, eps, one_hot_target_class):
    logits, _, end_points = inception(x, reuse=True)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                    logits,
                                                    label_smoothing=0.1,
                                                    weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                     end_points['AuxLogits'][:, 1:],
                                                     label_smoothing=0.1,
                                                     weights=0.4)
    x_adv = x - eps * tf.sign(tf.gradients(cross_entropy, x)[0])
    x_adv = tf.clip_by_value(x_adv, 0, 1.0)
    return tf.stop_gradient(x_adv)


def stepll_adversarial_images(x, eps):
    logits, _, _ = inception(x, reuse=True)
    least_likely_class = tf.argmin(logits, 1)
    one_hot_ll_class = tf.one_hot(least_likely_class, 1000)
    return step_target_class_adversarial_images(x, eps, one_hot_ll_class)


def stepllnoise_adversarial_images(x, eps):
    logits, _, _ = inception(x, reuse=True)
    least_likely_class = tf.argmin(logits, 1)
    one_hot_ll_class = tf.one_hot(least_likely_class, 1000)
    x_noise = x + eps / 2 * tf.sign(tf.random_normal(x.shape))
    return step_target_class_adversarial_images(x_noise, eps / 2,
                                                one_hot_ll_class)


# TODO
# 重要代码，获取激活分布8*8
layer_name = 'Mixed_7c'
num_class = 1000
conv_layer = end_point[layer_name]
pre_calss = tf.placeholder(tf.int32)
one_hot = tf.sparse_to_dense(pre_calss, [num_class], 1.0)
signal = tf.multiply(end_point['Logits'][:, 1:], one_hot)
loss = tf.reduce_mean(signal)
grads = tf.gradients(loss, conv_layer)[0]
norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))


def grad_cam(end_point, pre_calss_one_hot, layer_name='Mixed_7c'):
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
    # cam = tf.nn.relu(cam)
    resize_cam = tf.image.resize_images(cam, [299, 299])
    # resize_cam = resize_cam / tf.reduce_max(resize_cam)
    return resize_cam

    # cam = resize(cam, (299, 299))
    # Converting grayscale to 3-D
    # cam3 = np.expand_dims(cam, axis=2)
    # cam3 = np.tile(cam3, [1, 1, 3])
    # plt.imshow(cam3)
    # plt.show()


def get_count_IOU(rar, adv):
    rar_count = rar[rar == 1].size
    adv_count = adv[adv == 1].size
    sum = rar + adv
    IOU = sum[sum == 2].size / sum[sum != 0].size
    return rar_count, adv_count, IOU


x = tf.placeholder(tf.float32, (299, 299, 3))
x_hat = image  # our trainable adversarial input
assign_op = tf.assign(x_hat, x)
learning_rate = tf.placeholder(tf.float32, ())
y_hat = tf.placeholder(tf.int32, ())
labels = tf.one_hot(y_hat, 1000)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])
optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=[x_hat])
epsilon = tf.placeholder(tf.float32, ())
below = x - epsilon
above = x + epsilon
projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
with tf.control_dependencies([projected]):
    project_step = tf.assign(x_hat, projected)
FGSM_adv = stepll_adversarial_images(x_hat, 0.30)


def get_gard_cam(img_path, img_class, demo_target):
    demo_epsilon = 2.0 / 255.0
    demo_lr = 0.1
    demo_steps = 100
    img = PIL.Image.open(img_path).convert('RGB')
    big_dim = max(img.width, img.height)
    wide = img.width > img.height
    new_w = 299 if not wide else int(img.width * 299 / img.height)
    new_h = 299 if wide else int(img.height * 299 / img.width)
    img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
    img = (np.asarray(img) / 255.0).astype(np.float32)

    # 展示原分类图
    label_before = classify(img, correct_class=img_class)

    # 获取原图激活区域
    rar_gard_cam = grad_cam(img, img_class)

    # 显示被进攻后和的激活区域

    # initialization step
    """"""
    # FGSM攻击

    sess.run(assign_op, feed_dict={x: img})
    adv = sess.run(FGSM_adv)

    # for i in range(demo_steps):
    #     # gradient descent step
    #     _, loss_value = sess.run(
    #         [optim_step, loss],
    #         feed_dict={learning_rate: demo_lr, y_hat: demo_target})
    #     # project step
    #     sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
    #     if (i + 1) % 10 == 0:
    #         print('step %d, loss=%g' % (i + 1, loss_value))
    # adv = x_hat.eval()  # retrieve the adversarial example
    """"""

    # 展示攻击后的图像
    # classify(adv, correct_class=img_class)
    # 展示攻击后的图像的激活区域
    adv_gard_cam = grad_cam(adv, img_class)
    label_after = classify(adv, correct_class=img_class)

    return img, rar_gard_cam, adv_gard_cam, label_before, label_after


sess.graph.finalize()

if __name__ == '__main__':
    labels_file = 'imagenet_labels.txt'
    results_file = 'result/grad_result_fgsm_final.txt'

    if os.path.isfile (results_file):
        os.remove(results_file)
    with open(labels_file, 'r')as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            label_letter = line.split(' ')
            label_letter = label_letter[0]
            img_class = index
            demo_target = random.randint(0, 998)
            if demo_target == img_class:
                demo_target = random.randint(0, 998)
            dir_name = 'img_val/' + str(label_letter)
            for root, dirs, files in os.walk(dir_name):
                for file in files:
                    img_path = dir_name + '/' + file
                    try:
                        img, rar_gard_cam, adv_gard_cam, label_before, label_after = get_gard_cam(img_path, img_class,
                                                                                                  demo_target)
                    except Exception as e:
                        print(e)
                        continue
                    rar_count, adv_count, IOU = get_count_IOU(rar_gard_cam, adv_gard_cam)
                    print(index)
                    print(label_before, label_after, IOU)
                    with open(results_file, 'a') as f_w:
                        f_w.write(img_path + ' ' + str(img_class) + ' ' + str(demo_target)
                                  + ' ' + str(rar_count) + ' ' + str(adv_count) + ' '
                                  + str(label_before) + ' ' + str(label_after) + ' ' + str(IOU) + '\n')
                # plt.figure()
                # plt.subplot(1, 3, 1)
                # plt.imsave('img.png', img)
                # plt.subplot(1, 3, 2)
                # plt.imsave('rar_gard_cam' + '.png', rar_gard_cam)
                # plt.subplot(1, 3, 3)
                # plt.imsave('adv_gard_cam' + '.png', adv_gard_cam)
