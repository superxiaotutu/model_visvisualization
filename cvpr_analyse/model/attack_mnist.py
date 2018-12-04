import tensorflow as tf
import tensorflow.contrib.slim as slim
import cifarnet
import tensorflow.examples.tutorials.mnist.input_data as input_data
import LeNet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train = mnist.train.images
Y_train = mnist.train.labels

"""
总loss 记得用一下RCE
"""

_BATCH_SIZE = 1

X = tf.placeholder(tf.float32, [_BATCH_SIZE, 784])
X_adv = tf.placeholder(tf.float32, [_BATCH_SIZE, 784])
Y = tf.placeholder(tf.float32, [_BATCH_SIZE, 10])
keep_prop = tf.placeholder("float")




def bulid_Net(image, reuse=tf.AUTO_REUSE):
    image = tf.reshape(image, [-1, 28, 28, 1])
    with tf.variable_scope(name_or_scope='LeNet', reuse=reuse):
        arg_scope = LeNet.lenet_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_point = LeNet.lenet(image, 10, is_training=True, dropout_keep_prob=keep_prop)
            probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


def step_target_class_adversarial_images(x, eps, one_hot_target_class):
    logits, probs, end_points = bulid_Net(x)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                    logits,
                                                    label_smoothing=0.1,
                                                    weights=1.0)
    x_adv = x - eps * tf.sign(tf.gradients(cross_entropy, x)[0])
    x_adv = tf.clip_by_value(x_adv, 0, 1.0)
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
    weights = tf.tile(weights, [1, 7, 7, 1])
    pre_cam = tf.multiply(weights, conv_layer)
    cam = tf.reduce_sum(pre_cam, 3)
    cam = tf.expand_dims(cam, 3)
    """"""
    cam = tf.reshape(cam, [-1, 49])
    cam = tf.nn.softmax(cam)
    cam = tf.reshape(cam, [-1, 7, 7, 1])
    # cam = tf.nn.relu(cam)
    resize_cam = tf.image.resize_images(cam, [28, 28])
    resize_cam = resize_cam / tf.reduce_max(resize_cam)
    return resize_cam


def mask_image(img, grad_cam):
    img = tf.reshape(img, [-1, 28, 28, 1])
    cam_clip = tf.sign(tf.nn.relu(grad_cam - tf.reduce_mean(grad_cam)))
    reverse_cam = tf.abs(cam_clip - 1)
    return reverse_cam * img


adv_sample_get_op = stepll_adversarial_images(X, tf.random_uniform([1], 0, 0.3))
fixed_adv_sample_get_op = stepll_adversarial_images(X, 0.15)
NOISE_adv_sample_get_op = stepllnoise_adversarial_images(X, 0.15)

rar_logits, rar_probs, rar_end_point = bulid_Net(X)
adv_logits, adv_probs, adv_end_point = bulid_Net(X_adv)

rar_grad_cam = grad_cam(rar_end_point, Y)
adv_grad_cam = grad_cam(adv_end_point, Y)


correct_p = tf.equal(tf.argmax(rar_probs, 1), (tf.argmax(Y, 1)))
accuracy = tf.reduce_mean(tf.cast(correct_p, "float"))

modelname="model1_0.5_1_mard"
def restore():
    saver = tf.train.Saver()
    saver.restore(sess,modelname+"/model.ckpt")

restore()
sess.graph.finalize()
import numpy as np

print("进行测试集测试:")

arr1=[]
arr2=[]
arr3=[]
arr4=[]
arr5=[]
# 'result_my_mnist.npz','result_mard_mnist.npz',
arr_list=['result_mard_copy_mnist.npz']
for name in arr_list:
    ACC = 0
    adv_ACC = 0
    noise_adv_ACC = 0
    double_adv_ACC = 0
    arr=np.load(name)
    arr1=arr['arr_0']
    arr2=arr['arr_1']
    arr3=arr['arr_2']
    arr4=arr['arr_3']
    arr5=arr['arr_4']
    sess.graph.finalize()
    a=0
    for adv_sample,noise_adv_sample,double_adv_sample,ori_sample,label in zip(arr1,arr2,arr3,arr4,arr5):
        a+=1
        noise_adv_ACC += sess.run(accuracy, feed_dict={X: [noise_adv_sample], Y: [label], keep_prop: 1.0})
        ACC += sess.run(accuracy, feed_dict={X: [ori_sample], Y: [label], keep_prop: 1.0})
        adv_ACC += sess.run(accuracy, feed_dict={X: [adv_sample], Y: [label], keep_prop: 1.0})
        double_adv_ACC += sess.run(accuracy, feed_dict={X: [double_adv_sample], Y: [label], keep_prop: 1.0})
        print(a)
    print(ACC / (10000 // _BATCH_SIZE))
    print(adv_ACC / (10000 // _BATCH_SIZE))
    print(noise_adv_ACC / (10000 // _BATCH_SIZE))
    print(double_adv_ACC / (10000 // _BATCH_SIZE))

    with open('mnist_attack','a') as f:
        f.write("\n"+modelname+name+str(ACC / (10000 // _BATCH_SIZE)))
        f.write(str(adv_ACC / (10000 // _BATCH_SIZE)))
        f.write(str(noise_adv_ACC / (10000 // _BATCH_SIZE)))
        f.write(str(double_adv_ACC / (10000 // _BATCH_SIZE))+"\n")