import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from skimage.transform import resize
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sess = tf.InteractiveSession()

image = tf.Variable(tf.zeros((299, 299, 3)))

def inception(image, reuse):
    preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, end_point = nets.inception.inception_v3(preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:,1:] # ignore background class
        probs = tf.nn.softmax(logits) # probabilities
    return logits, probs, end_point

logits, probs, end_point = inception(image, reuse=False)

restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV3/')
]
saver = tf.train.Saver(restore_vars)
saver.restore(sess, "inception_v3.ckpt")

import json
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
plt.switch_backend('agg')

imagenet_json= ('imagenet.json')
with open(imagenet_json) as f:
    imagenet_labels = json.load(f)


def classify(img, correct_class=None, target_class=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.sca(ax1)
    p = sess.run(probs, feed_dict={image: img})[0]
    ax1.imshow(img)
    fig.sca(ax1)


    topk = list(p.argsort()[-10:][::-1])
    topprobs = p[topk]
    print(np.argmax(p))
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


def grad_cam(x, end_point , pre_calss, layer_name='Mixed_7c', num_class=1000):
    conv_layer = end_point[layer_name]
    one_hot = tf.sparse_to_dense(pre_calss, [num_class], 1.0)
    signal = tf.multiply(end_point['Logits'][:, 1:], one_hot)
    loss = tf.reduce_mean(signal)
    grads = tf.gradients(loss, conv_layer)[0]
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))
    output, grads_val = sess.run([conv_layer, norm_grads], feed_dict={image: x})
    output = output[0]
    grads_val = grads_val[0]
    weights = np.mean(grads_val, axis=(0, 1))  # [512]
    cam = np.ones(output.shape[0: 2], dtype=np.float32)  # [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    # np.set_printoptions(threshold='nan')

    """"""
    cam = cam - np.mean(cam)
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    cam = np.resize(cam, (20,20))
    # cam = cam - np.mean(cam)
    print(np.sign(cam))
    # cam3 = np.expand_dims(cam, axis=2)
    # cam3 = np.tile(cam3, [1, 1, 3])
    # plt.imsave('cam3.png',cam3)
    # plt.imshow(cam3)
    # plt.show()
    return np.sign(cam)

import PIL
import numpy as np

img_path = "ILSVRC2012_val_00000293.JPEG"
xml_path="ILSVRC2012_val_00000293.xml"
img_class = 283
img = PIL.Image.open(img_path)
big_dim = max(img.width, img.height)
wide = img.width > img.height
new_w = 299 if not wide else int(img.width * 299 / img.height)
new_h = 299 if wide else int(img.height * 299 / img.width)
img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
img = (np.asarray(img) / 255.0).astype(np.float32)

classify(img, correct_class=img_class)
rar_gard_cam = grad_cam(img, end_point, img_class)


# Initialization
x = tf.placeholder(tf.float32, (299, 299, 3))

x_hat = image # our trainable adversarial input
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

demo_epsilon = 2.0 / 255.0  # a really small perturbation
demo_lr = 0.1
demo_steps = 100
demo_target = 123


FGSM_adv = stepll_adversarial_images(x_hat, 0.30)


# initialization step
sess.run(assign_op, feed_dict={x: img})


""""""
adv = sess.run(FGSM_adv)

# # # projected gradient descent
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

classify(adv, correct_class=img_class)
adv_gard_cam = grad_cam(adv, end_point, img_class)

plt.imsave('img.png',img)
plt.imsave('rar_gard_cam.png',rar_gard_cam)
plt.imsave('adv_gard_cam.png',adv_gard_cam)
