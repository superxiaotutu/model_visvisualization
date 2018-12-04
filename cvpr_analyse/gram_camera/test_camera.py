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
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib.pyplot as plt

# plt.switch_backend('agg')


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
    p = sess.run(probs, feed_dict={image: img})[0]
    return np.argmax(p)


# TODO
# 重要代码，获取激活分布8*8
layer_name='Mixed_7c'
num_class=1000
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
    # cam=np.exp(cam) / np.sum(np.exp(cam), axis=0)
    # cam=cam/np.max(cam)
    # cam3 = np.expand_dims(cam, axis=2)
    # cam3 = np.tile(cam3, [1, 1, 3])

    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam3 = cv2.resize(cam, (299, 299))

    # cam3=np.expand_dims(cam,axis=2)
    # cam=np.tile(cam3,[1,1,3])
    # cam = resize(cam, (299, 299,3))


    return cam3


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

    # 展示原分类图

    # 获取原图激活区域
    rar_gard_cam = grad_cam(img, img_class)

    # 显示被进攻后和的激活区域



    """"""
    # 展示攻击后的图像
    # 展示攻击后的图像的激活区域

    return img, rar_gard_cam

def show_img(file_name,img,rar,adv):
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    img = cv2.resize(img, (299, 299))
    img = img.astype(float)
    img /= img.max()
    rar = cv2.applyColorMap(np.uint8(255 * rar), cv2.COLORMAP_JET)
    rar = cv2.cvtColor(rar, cv2.COLOR_BGR2RGB)
    alpha = 0.0072
    rar = img + alpha * rar
    rar /= rar.max()
    # Display and save
    plt.imshow(rar)
    plt.subplot(1, 3, 3)
    adv = cv2.applyColorMap(np.uint8(255 * adv), cv2.COLORMAP_JET)
    adv = cv2.cvtColor(adv, cv2.COLOR_BGR2RGB)
    alpha = 0.0072
    adv = img + alpha * adv
    adv /= adv.max()
    plt.imshow(adv)
    plt.savefig(file_name)
    plt.close()
sess.graph.finalize()
def get_label_name(index):
    with open('imagenet_labels.txt','r',encoding='utf8')as f:
        data=f.read(index+1)
    return data
if __name__ == '__main__':
    print(get_label_name(0))
    # for r,d,f in os.walk('img_val/n01440764'):
    #     for file in f:
    #         imgs=[]
    #         labels_file = 'imagenet_labels.txt'
    #         results_file = 'result.txt'
    #         print('img_val/n01440764/'+file)
    #         img, cam3 = get_gard_cam('img_val/n01440764/'+file, 0)
    #         show_img(img,cam3,cam3)