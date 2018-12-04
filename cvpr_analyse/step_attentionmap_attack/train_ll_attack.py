import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from urllib.request import urlretrieve
import json
import matplotlib.pyplot as plt
import PIL
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
plt.switch_backend('agg')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
y_hat = tf.placeholder(tf.int32, ())
labels = tf.one_hot(y_hat, 1000)

x = tf.placeholder(tf.float32, (1, 299, 299, 3))
x_rar = tf.placeholder(tf.float32, (1, 299, 299, 3))
x_adv = tf.Variable(tf.zeros([1, 299, 299, 3]))

_POOL_NAME = 'Mixed_7c'
_POOL_SIZE = 8
_MODEL_END = 'Logits'


def inception(image, reuse):
    preprocessed = tf.multiply(tf.subtract(image, 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, end_point = nets.inception.inception_v3(preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


def grad_cam(end_point, pre_calss_one_hot):
    conv_layer = end_point[_POOL_NAME]
    signal = tf.multiply(end_point[_MODEL_END][:, 1:], pre_calss_one_hot)
    loss = tf.reduce_mean(signal, 1)
    grads = tf.gradients(loss, conv_layer)[0]
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))
    weights = tf.reduce_mean(norm_grads, axis=(1, 2))
    weights = tf.expand_dims(weights, 1)
    weights = tf.expand_dims(weights, 1)
    weights = tf.tile(weights, [1, _POOL_SIZE, _POOL_SIZE, 1])
    pre_cam = tf.multiply(weights, conv_layer)
    cam = tf.reduce_sum(pre_cam, 3)
    cam = tf.expand_dims(cam, 3)
    cam = tf.nn.relu(cam)
    resize_cam = tf.image.resize_images(cam, [299, 299])
    resize_cam = resize_cam / tf.reduce_max(resize_cam)
    return resize_cam


def cal_IOU(rar_map, adv_map):
    clip_rar = tf.sign(tf.nn.relu(rar_map - tf.reduce_max(rar_map) * 0.5))
    clip_adv = tf.sign(tf.nn.relu(adv_map - tf.reduce_max(rar_map) * 0.5))

    tt_tmp = clip_rar + clip_adv
    total_clip = tf.sign(tt_tmp)

    b_tmp = tf.nn.relu(clip_rar + clip_adv - 1)
    bing_clip = tf.sign(b_tmp)

    iou = tf.reduce_sum(bing_clip, [1, 2, 3]) / tf.reduce_sum(total_clip, [1, 2, 3])
    return iou


def g_loss(rar_map, adv_map):
    grad_cam_loss = tf.reduce_sum(tf.pow(rar_map - adv_map, 2))
    return grad_cam_loss


def sign_gloss(rar_map, adv_map):
    clip_rar = tf.sign(tf.nn.relu(rar_map - tf.reduce_max(rar_map) * 1))
    clip_rar = tf.reshape(clip_rar, [-1, 299 * 299])
    flatten_rar_amp = tf.reshape(rar_map, [-1, 299 * 299])
    flatten_adv_map = tf.reshape(adv_map, [-1, 299 * 299])

    gloss = tf.reduce_mean(tf.abs(clip_rar - flatten_adv_map))
    # gloss = tf.reduce_mean(tf.abs((1-flatten_rar_amp)-flatten_adv_map))

    # closs
    ARGMIN = tf.argmin(probs, 1)
    LL = tf.one_hot(ARGMIN, 1000)
    closs = tf.losses.softmax_cross_entropy(LL, adv_logits)
    # closs = 0
    # return closs
    return closs


logits, probs, end_point = inception(x, reuse=tf.AUTO_REUSE)
rar_logits, rar_probs, rar_end_point = inception(x_rar, reuse=tf.AUTO_REUSE)
adv_logits, adv_probs, adv_end_point = inception(x_adv, reuse=tf.AUTO_REUSE)

restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV3/')
    ]
saver = tf.train.Saver(restore_vars)
saver.restore(sess, "../gram_camera/inception_v3.ckpt")

imagenet_json='imagenet.json'
with open(imagenet_json) as f:
    imagenet_labels = json.load(f)


def classify(img, correct_class=None, target_class=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.sca(ax1)
    p = sess.run(rar_probs, feed_dict={x_rar: img})[0]
    ax1.imshow(img)
    fig.sca(ax1)

    topk = list(p.argsort()[-10:][::-1])
    topprobs = p[topk]
    barlist = ax2.bar(range(10), topprobs)
    if target_class in topk:
        barlist[topk.index(target_class)].set_color('r')
    if correct_class in topk:
        barlist[topk.index(correct_class)].set_color('g')
    plt.sca(ax2)
    plt.ylim([0, 1.1])
    plt.xticks(range(10),
               [imagenet_labels[i][:15] for i in topk],
               rotation='vertical')
    fig.subplots_adjust(bottom=0.2)
    plt.show()


def load(img_path):
    img = PIL.Image.open(img_path)
    big_dim = max(img.width, img.height)
    wide = img.width > img.height
    new_w = 299 if not wide else int(img.width * 299 / img.height)
    new_h = 299 if wide else int(img.height * 299 / img.width)
    img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
    img = (np.asarray(img) / 255.0).astype(np.float32)
    img = img[:, :, :3]

    return img

rar_grad_cam = grad_cam(rar_end_point, labels)
adv_grad_cam = grad_cam(adv_end_point, labels)

get_random = tf.sign(tf.random_normal([299, 299, 3])) * 8 / 255

get_iou_op = cal_IOU(rar_grad_cam, adv_grad_cam)
gloss = g_loss(rar_grad_cam, adv_grad_cam)

assign_op = tf.assign(x_adv, x)

# 自动更新
get_sign_gloss = sign_gloss(rar_grad_cam, adv_grad_cam)
train_op = tf.train.GradientDescentOptimizer(1).minimize(get_sign_gloss, var_list=[x_adv])
# 手动更新
grad_sgloss = tf.gradients(get_sign_gloss, x_adv)[0]
g_assign = tf.assign(x_adv, x_adv + tf.sign(grad_sgloss) * 1 / 255)
# project
epsilon = 4 / 255
below = x - epsilon
above = x + epsilon
projected = tf.clip_by_value(tf.clip_by_value(x_adv, below, above), 0, 1)
with tf.control_dependencies([projected]):
    project_step = tf.assign(x_adv, projected)

correct = tf.equal(tf.argmax(rar_probs, 1), (tf.argmax(adv_probs, 1)))

labels_file = 'imagenet_labels.txt'
results_file = 'll_result.txt'
import os
correct_count=0
wrong_count=0
count=0
sess.graph.finalize()
with open(labels_file, 'r')as f:
    lines = f.readlines()
    if os.path.isfile(results_file):
        os.remove(results_file)
    for index, line in enumerate(lines):
        label_letter = line.split(' ')
        label_letter = label_letter[0]
        img_class = index
        dir_name = '../gram_camera/img_val/' + str(label_letter)
        for root, dirs, files in os.walk(dir_name):
            for index2, file in enumerate(files):
                    img_path = dir_name + '/' + file
                    print(img_class)
                    img = load(img_path)
                    rar_img = img
                    sess.run(assign_op, feed_dict={x: [img]})
                    # print(np.argmax(sess.run(probs, feed_dict={x: [img]})))


                    for i in range(1):
                        sess.run(train_op, feed_dict={x: [img], x_rar: [rar_img], y_hat: img_class})
                        sess.run(project_step, feed_dict={x: [rar_img]})
                        result = (sess.run(correct, feed_dict={x: [img], x_rar: [rar_img], y_hat: img_class}))
                        if result[0]:
                            correct_count+=1
                        else:
                            wrong_count += 1
                        count+=1
                        print(correct_count,wrong_count,count)
                        with open(results_file, 'a')as f:
                            f.write(str(result[0]) + '\n')

    f.write(str(correct_count)+" "+str(wrong_count)+" "+str(count) + '\n')

# print(np.argmax(sess.run(adv_probs)), sess.run(get_iou_op, feed_dict={x_rar: [img], y_hat: img_class}))
#     adv_map = sess.run(adv_grad_cam, feed_dict={y_hat: img_class})
#     adv_map = np.reshape(adv_map, [299, 299])
#     plt.imshow(adv_map)
#     plt.show()
#
# adv = sess.run(x_adv)
# adv = np.reshape(adv, [299, 299, 3])
# plt.imshow(adv)
# plt.show()
