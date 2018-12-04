import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.examples.tutorials.mnist.input_data as input_data
import LeNet
import numpy as np

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


def restore():
    saver = tf.train.Saver()
    saver.restore(sess, "model1_0.5_1_my_copy/model.ckpt")


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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

restore()
sess.graph.finalize()
print("进行测试集测试:")
ACC = 0
adv_ACC = 0
noise_adv_ACC = 0
double_adv_ACC = 0
arr1=[]
arr2=[]
arr3=[]
arr4=[]
arr5=[]
sess.graph.finalize()
for i in range(10000):
    print(i)
    testbatch = mnist.test.next_batch(1)
    adv_sample = sess.run(fixed_adv_sample_get_op, feed_dict={X: testbatch[0], keep_prop: 1.0})
    noise_adv_sample = sess.run(NOISE_adv_sample_get_op, feed_dict={X: testbatch[0], keep_prop: 1.0})
    double_adv_sample = sess.run(fixed_adv_sample_get_op, feed_dict={X: adv_sample, keep_prop: 1.0})
    arr1.append(adv_sample[0])
    arr2.append(noise_adv_sample[0])
    arr3.append(double_adv_sample[0])
    arr4.append(testbatch[0][0])
    arr5.append(testbatch[1][0])

np.savez('result_my_copy_mnist',arr1,arr2,arr3,arr4,arr5)

