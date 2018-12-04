


import os
import random

import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from skimage.transform import resize
import PIL
import numpy as np
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt

plt.switch_backend('agg')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
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
def classify(img):
    p = sess.run(probs, feed_dict={image: img})[0]
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
 のlayer_name = 'Mixed_7c'
num_class = 1000
conv_layer = end_point[layer_name]
pre_calss = tf.placeholder(tf.int32)
one_hot = tf.sparse_to_dense(pre_calss, [num_class], 1.0)
signal = tf.multiply(end_point['Logits'][:, 1:], one_hot)
loss = tf.reduce_mean(signal)
grads = tf.gradients(loss, conv_layer)[0]
norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))


def grad_cam(x, class_num):
    output, grads_val = sess.run([conv_layer, norm_grads], feed_dict={image: x, pre_calss: class_num})
    output = output[0]
    grads_val = grads_val[0]
    weights = np.mean(grads_val, axis=(0, 1))  # [512]
    cam = np.ones(output.shape[0: 2], dtype=np.float32)  # [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU

    """"""
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam3 = np.expand_dims(cam, axis=2)
    cam = np.tile(cam3, [1, 1, 3])
    cam = resize(cam, (299, 299, 3))
    return cam


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


def get_gard_cam(img_path, img_class):
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
    img = img[:, :, :3]
    # 展示原分类图
    label_before = classify(img, )

    # 获取原图激活区域
    rar_gard_cam = grad_cam(img, img_class)

    # 显示被进攻后和的激活区域

    sess.run(assign_op, feed_dict={x: img})
    adv = sess.run(FGSM_adv)

    """"""

    # 展示攻击后的图像
    label_after = classify(adv)
    # 展示攻击后的图像的激活区域
    adv_gard_cam = grad_cam(adv, img_class)

    return img, rar_gard_cam, adv_gard_cam, label_before, label_after


sess.graph.finalize()


def show_img(file_name, img, rar, adv,ground_truth):
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis('off')
    # plt.title(rar_label)
    plt.subplot(1, 3, 2)
    img = cv2.resize(img, (299, 299))
    img = img.astype(float)
    img /= img.max()
    rar = cv2.applyColorMap(np.uint8(255 * rar), cv2.COLORMAP_JET)
    rar = cv2.cvtColor(rar, cv2.COLOR_BGR2RGB)
    alpha = 0.0072
    rar = img + alpha * rar
    rar /= rar.max()
    plt.imshow(rar)
    plt.axis('off')
    # plt.title(rar_label)
    plt.subplot(1, 3, 3)
    # adv = cv2.applyColorMap(np.uint8(255 * adv), cv2.COLORMAP_JET)
    adv = cv2.applyColorMap(np.uint8(255 * ground_truth), cv2.COLORMAP_JET)
    adv = cv2.cvtColor(adv, cv2.COLOR_BGR2RGB)
    alpha = 0.0072
    adv = img + alpha * adv
    adv /= adv.max()
    # plt.imshow(adv)
    plt.imshow(adv)
    plt.axis('off')
    plt.savefig(file_name)
    plt.close()

def get_label_name(index):
    with open('imagenet.json') as f:
        imagenet_labels = json.load(f)
        label = imagenet_labels[index]
    return label


if __name__ == '__main__':
