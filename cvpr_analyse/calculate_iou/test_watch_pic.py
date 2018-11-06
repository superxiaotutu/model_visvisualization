import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import numpy as np
import os
import PIL
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

_BATCH_SIZE = 1
X = tf.placeholder(tf.float32, [_BATCH_SIZE, 299, 299, 3])
Y = tf.placeholder(tf.int32, [_BATCH_SIZE])

sess = tf.InteractiveSession()


def inception(image, reuse=tf.AUTO_REUSE):
    preprocessed = tf.multiply(tf.subtract(image, 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, end_point = nets.inception.inception_v3(preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


def step_target_class_adversarial_images(x, eps, one_hot_target_class):
    _, logits, end_points = inception(x)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                    logits,
                                                    label_smoothing=0.1,
                                                    weights=1.0)
    x_adv = x - eps * tf.sign(tf.gradients(cross_entropy, x)[0])
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
    return tf.stop_gradient(x_adv)


def stepll_adversarial_images(x, eps):
    _, logits, end_points = inception(x)
    least_likely_class = tf.argmin(logits, 1)
    one_hot_ll_class = tf.one_hot(least_likely_class, 1000)
    return step_target_class_adversarial_images(x, eps, one_hot_ll_class)


def grad_cam(end_point, Y, layer_name='Mixed_7c'):
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
    # cam = tf.nn.relu(cam)
    resize_cam = tf.image.resize_images(cam, [299, 299])
    '''新加sign'''
    resize_cam = resize_cam - tf.reduce_mean(resize_cam)
    resize_cam = tf.nn.relu(resize_cam)
    return tf.sign(resize_cam)


fixed_adv_sample_get_op = stepll_adversarial_images(X, 0.15)

rar_logits, rar_probs, rar_end_point = inception(X)
adv_logits, adv_probs, adv_end_point = inception(fixed_adv_sample_get_op)

is_attack = tf.equal(tf.argmax(rar_probs, 1), (tf.argmax(adv_probs, 1)))

rar_grad_cam = grad_cam(rar_end_point, Y)
adv_grad_cam = grad_cam(adv_end_point, Y)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "inception_v3.ckpt")


def load_img(path):
    I = PIL.Image.open(path).convert('RGB')
    I = I.resize((299, 299)).crop((0, 0, 299, 299))
    I = (np.asarray(I) / 255.0).astype(np.float32)
    return I[:, :, 0:3]


def get_gound_truth(label_txt):
    fp = open(label_txt)
    ground_truth = np.zeros((299, 299))
    label = 0
    for p in fp:
        if '<size>' in p:
            width = int(next(fp).split('>')[1].split('<')[0])
            height = int(next(fp).split('>')[1].split('<')[0])

        if '<object>' in p:
            label = next(fp).split('>')[1].split('<')[0]
        if '<bndbox>' in p:
            xmin = int(next(fp).split('>')[1].split('<')[0])
            ymin = int(next(fp).split('>')[1].split('<')[0])
            xmax = int(next(fp).split('>')[1].split('<')[0])
            ymax = int(next(fp).split('>')[1].split('<')[0])
            matrix = [int(xmin / width * 299), int(ymin / height * 299), int(xmax / width * 299),
                      int(ymax / height * 299)]
            ground_truth[ matrix[0]:matrix[2],matrix[1]:matrix[3]] = 1

    return ground_truth


def get_iou(rar, adv, ground_truth):
    if ground_truth[ground_truth == 1].size == 0:
        return rar, adv, ground_truth, False
    rar_count = rar[rar == 1].size
    adv_count = adv[adv == 1].size
    ground_count = ground_truth[ground_truth == 1].size
    rar_sum = rar + ground_truth
    adv_sum = adv + ground_truth
    rar_IOU = rar_sum[rar_sum == 2].size / ground_count
    adv_IOU = adv_sum[adv_sum == 2].size / ground_count
    return rar_count, adv_count, rar_IOU, adv_IOU


if __name__ == '__main__':
    imgs=[]
    labels=[0]
    imgs.append(load_img('adv.png'))
    # gt=get_gound_truth('0000293.xml')
    rar_maps, adv_maps = sess.run([rar_grad_cam, adv_grad_cam], feed_dict={X: imgs, Y: labels})
    # rar_maps = np.reshape(rar_maps, (_BATCH_SIZE, 299, 299))
    # adv_maps = np.reshape(adv_maps, (_BATCH_SIZE, 299, 299))
    j=0
    # rar_count, adv_count, rar_g_iou, adv_g_iou = get_iou(rar_maps[j], adv_maps[j], gt)

    plt.imshow(imgs[0])
    plt.show()

    plt.imshow(rar_maps[0])
    plt.show()
    plt.imshow(adv_maps[0])
    plt.show()
    # plt.imshow(gt)
    # plt.show()

    # plt.imshow(np.resize(imgs[0],(299,299))*gt)
    plt.show()

