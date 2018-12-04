import tensorflow as tf
import tensorflow.contrib.slim as slim
import cifarnet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name='model/model_step2_8.0_4.0_1.0'
dir_name='step2loss/'
import numpy as np
# import  matplotlib.pyplot as plt
# import cv2
_BATCH_SIZE = 30
lr = 0.0001

X = tf.placeholder(tf.float32, [_BATCH_SIZE, 32, 32, 3])
X_adv = tf.placeholder(tf.float32, [_BATCH_SIZE, 32, 32, 3])
Y = tf.placeholder(tf.float32, [_BATCH_SIZE, 10])


def bulid_Net(image, reuse=tf.AUTO_REUSE):
    image = tf.reshape(image, [-1, 32, 32, 3])
    with tf.variable_scope(name_or_scope='CifarNet', reuse=reuse):
        arg_scope = cifarnet.cifarnet_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_point = cifarnet.cifarnet(image, 10, is_training=False)
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


def grad_cam(end_point, pre_calss_one_hot, layer_name='pool2'):
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
    resize_cam = tf.image.resize_images(cam, [32, 32])
    # resize_cam = resize_cam / tf.reduce_max(resize_cam)
    return resize_cam,cam


def mask_image(img, grad_cam):
    img = tf.reshape(img, [-1, 32, 32, 3])
    cam_clip = tf.sign(tf.nn.relu(grad_cam - tf.reduce_mean(grad_cam)))
    reverse_cam = tf.abs(cam_clip - 1)
    return reverse_cam * img


def restore():
    saver = tf.train.Saver()
    saver.restore(sess, model_name+"/model.ckpt")

# our 0.05


adv_sample_get_op = stepll_adversarial_images(X, tf.random_uniform([1], 0, 0.3))
NOISE_adv_sample_get_op = stepllnoise_adversarial_images(X, 0.15)





fixed_adv_sample_get_op = stepll_adversarial_images(X, 0.15)

rar_logits, rar_probs, rar_end_point = bulid_Net(X)
adv_logits, adv_probs, adv_end_point = bulid_Net(X_adv)

rar_grad_cam, rar_rarcam = grad_cam(rar_end_point, Y)
adv_grad_cam, adv_rarcam = grad_cam(adv_end_point, Y)


mask_X = mask_image(X, rar_grad_cam)
mask_X_adv = mask_image(X_adv, adv_grad_cam)

correct_p = tf.equal(tf.argmax(rar_probs, 1), (tf.argmax(Y, 1)))
rar_labels=tf.argmax(rar_probs, 1)
adv_labels=tf.argmax(adv_probs, 1)

accuracy = tf.reduce_mean(tf.cast(correct_p, "float"))

tf.summary.scalar('accuracy', accuracy)

JUDGE = tf.equal(tf.argmax(rar_probs, 1), (tf.argmax(adv_probs, 1)))



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

test_batchs = test_read_and_decode(test_filename, _BATCH_SIZE)
coord = tf.train.Coordinator()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

threads = tf.train.start_queue_runners(coord=coord, sess=sess)


restore()


# def show(img,rar,adv):
    # plt.figure()
    #
    # rar_mask[rar_mask != 2] = 0
    # rar_mask = cv2.applyColorMap(np.uint8(255 * rar_mask), cv2.COLORMAP_JET)
    # rar_mask = cv2.cvtColor(rar_mask, cv2.COLOR_BGR2RGB)
    # img = img.astype(float)
    # img /= img.max()
    # alpha = 2
    # rar_img = img + alpha * rar_mask
    #
    # adv_mask[adv_mask != 2] = 0
    # adv_mask = cv2.applyColorMap(np.uint8(255 * adv_mask), cv2.COLORMAP_JET)
    # adv_mask = cv2.cvtColor(adv_mask, cv2.COLOR_BGR2RGB)
    # adv_img = img + alpha * adv_mask
    #
    # plt.subplot(1, 5, 1)
    # plt.imshow(img)
    # plt.subplot(1, 5, 2)
    # plt.imshow(rar_mask)
    #
    # # plt.imshow(mark_boundaries(images[j] / 2 + 0.5, rar_mask))
    # # np.save('a.npy', mark_boundaries(images[j] / 2 + 0.5, rar_mask))
    # plt.subplot(1, 5, 3)
    # plt.imshow(adv_img)
    #
    # # plt.imshow(mark_boundaries(images[j] / 2 + 0.5, adv_mask))
    # plt.subplot(1, 5, 4)
    # plt.imshow(rar_img)
    # plt.subplot(1, 5, 5)
    # plt.imshow(adv_img)
    # plt.savefig(
    #     "image/" + str(rar_label) + "_" + str(adv_label) + "_" + str(num_features) + "_" + str(
    #         j) + '.png')

if  not  os.path.isdir(dir_name):
    # os.removedirs(dir_name)
    os.mkdir(dir_name)
for i in range(10000//_BATCH_SIZE):
    test_batch = sess.run(test_batchs)
    correct_ps=sess.run(accuracy,feed_dict={X: test_batch[0],  Y: test_batch[1]})
    print(correct_ps)
    _rar_labels = sess.run(rar_labels, feed_dict={X: test_batch[0], Y: test_batch[1]})
    adv_sample = sess.run(fixed_adv_sample_get_op, feed_dict={X: test_batch[0]})
    _adv_labels = sess.run(adv_labels, feed_dict={X_adv: adv_sample, Y: test_batch[1]})
    rar_maps, adv_maps = sess.run([rar_rarcam, adv_rarcam],
                                  feed_dict={X: test_batch[0], X_adv: adv_sample, Y: test_batch[1]})

    print(i)
    for j in range(_BATCH_SIZE):
        true_label=np.argmax(test_batch[1][j])
        print(true_label,_rar_labels[j])
        if true_label==_rar_labels[j]:
            rar_label=_rar_labels[j]
            adv_label=_adv_labels[j]
            np.savez(
                dir_name + str(rar_label) + "_" + str(adv_label) + "_" + str(i)+str(
                    j),test_batch[0][j], rar_maps[j], adv_maps[j])

        # plt.figure()
        # plt.subplot(1, 3, 1)
        # plt.imshow(test_batch[0][j])
        # plt.subplot(1, 3, 2)
        # rar_cam=np.reshape(rar_maps[j],[8,8])
        # plt.imshow(rar_cam)
        # plt.subplot(1, 3, 3)
        # adv_cam=np.reshape(adv_maps[j],[8,8])
        # plt.imshow(adv_cam)
        # plt.show()

