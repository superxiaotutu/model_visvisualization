import os
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cifarnet
import numpy as np
import sys

a, b, c, cuda = 8, 4, 1, '2,3'

print(a, b, c, cuda)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda

a, b, c, d = float(a), float(b), float(c), 1
_BATCH_SIZE = 50
lr = 0.0001
attack_step = 0.15
top_key = 32
key = 0.37

IOUS = tf.placeholder(tf.float32, [_BATCH_SIZE])
X = tf.placeholder(tf.float32, [_BATCH_SIZE, 32, 32, 3])
X_adv = tf.placeholder(tf.float32, [_BATCH_SIZE, 32, 32, 3])
Y = tf.placeholder(tf.float32, [_BATCH_SIZE, 10])
keep_prop = tf.placeholder("float")


def bulid_Net(image, reuse=tf.AUTO_REUSE):
    image = tf.reshape(image, [-1, 32, 32, 3])
    with tf.variable_scope(name_or_scope='CifarNet', reuse=reuse):
        arg_scope = cifarnet.cifarnet_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_point = cifarnet.cifarnet(image, 10, is_training=False, dropout_keep_prob=keep_prop)
            probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


def step_target_class_adversarial_images(x, eps, one_hot_target_class):
    logits, probs, end_points = bulid_Net(x)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                    logits,
                                                    label_smoothing=0.1,
                                                    weights=1.0)

    IOUS_ = IOUS - key
    IOUS_ = tf.nn.relu(IOUS_)
    IOUS_ = tf.sign(IOUS_)
    IOUS_ = tf.expand_dims(IOUS_, 1)
    IOUS_ = tf.expand_dims(IOUS_, 1)
    IOUS_ = tf.expand_dims(IOUS_, 1)
    IOUS_ = tf.tile(IOUS_, [1, 32, 32, 3])
    x_adv = x - eps * tf.sign(tf.gradients(cross_entropy, x)[0]) * IOUS_
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
    resize_cam = tf.sign(resize_cam)
    # resize_cam = resize_cam / tf.reduce_max(resize_cam)
    # cam = cam - tf.reduce_mean(cam)
    # cam = tf.nn.relu(cam)
    # cam = tf.sign(cam)

    return resize_cam, cam


def mask_image(img, grad_cam):
    img = tf.reshape(img, [-1, 32, 32, 3])
    cam_clip = tf.sign(tf.nn.relu(grad_cam - tf.reduce_mean(grad_cam)))
    reverse_cam = tf.abs(cam_clip - 1)
    return reverse_cam * img


def save():
    saver = tf.train.Saver()
    saver.save(sess, "equal_iou" + str(key) + "_" + str(a) + '_' + str(b) + '_' + str(c) + "/model.ckpt")


def restore():
    saver = tf.train.Saver()
    saver.restore(sess, "model_iou_" + str(key) + "_" + str(a) + '_' + str(b) + '_' + str(c) + "/model.ckpt")


def blance_rar_adv(x):
    x = tf.reshape(x, [_BATCH_SIZE, 8, 8])
    threshold = tf.reshape(x, [_BATCH_SIZE, 64])
    threshold = tf.nn.top_k(threshold, k=top_key)
    threshold = threshold.values[:, -1]
    threshold = tf.expand_dims(threshold, 1)
    threshold = tf.expand_dims(threshold, 1)
    threshold = tf.tile(threshold, [1, 8, 8])
    x = x - threshold
    x = tf.nn.relu(x)
    x = tf.sign(x)
    return x


def get_rar_adv_iou(rar, adv):
    rar = blance_rar_adv(rar)
    adv = blance_rar_adv(adv)
    sum = tf.add(rar, adv)
    count_no_0 = tf.count_nonzero(sum, [1, 2])
    count_no_0 = tf.cast(count_no_0, tf.float32)
    temp_sub_2 = tf.subtract(sum, constant_2)
    temp_sub_2_nonzero = tf.count_nonzero(temp_sub_2, [1, 2])
    count_2 = tf.subtract(constant_64, temp_sub_2_nonzero)
    count_2 = tf.cast(count_2, tf.float32)
    IOU = tf.divide(count_2, count_no_0)
    return IOU

def show_rar_adv(rar,adv):
    rar = blance_rar_adv(rar)
    adv = blance_rar_adv(adv)
    return rar,adv
constant_2 = tf.constant(2, dtype=tf.float32)
constant_64 = tf.constant(64, dtype=tf.int64)


# 主要分类节点
NOISE_adv_sample_get_op = stepllnoise_adversarial_images(X, attack_step)
fixed_adv_sample_get_op = stepll_adversarial_images(X, attack_step)
rar_logits, rar_probs, rar_end_point = bulid_Net(X)
adv_logits, adv_probs, adv_end_point = bulid_Net(X_adv)
# gradcam并判断iou
is_correct = tf.equal(tf.argmax(rar_probs, 1), (tf.argmax(Y, 1)))
is_defense = tf.equal(tf.argmax(rar_probs, 1), (tf.argmax(adv_probs, 1)))
rar_grad_cam, rar_cam = grad_cam(rar_end_point, Y)
adv_grad_cam, adv_cam = grad_cam(adv_end_point, Y)
constant_03 = tf.constant(0.3, dtype=tf.float32)
rar_adv_ious = get_rar_adv_iou(rar_cam, adv_cam)
show_rar_advs= show_rar_adv(rar_cam,adv_cam)

with tf.name_scope("classifition_loss"):
    loss_1 = slim.losses.softmax_cross_entropy(rar_logits, Y)
    loss_2 = slim.losses.softmax_cross_entropy(adv_logits, Y)
    regularization = tf.add_n(slim.losses.get_regularization_losses())
    classification_loss = loss_1 + loss_2 + regularization

with tf.name_scope("grad_cam_loss"):
    # grad_cam_loss = tf.reduce_sum(tf.abs(rar_grad_cam - adv_grad_cam))
    grad_cam_loss = tf.reduce_sum(tf.square(rar_grad_cam - adv_grad_cam))

mask_X = mask_image(X, rar_grad_cam)
mask_X_adv = mask_image(X_adv, adv_grad_cam)

Mrar_logits, Mrar_probs, Mrar_end_point = bulid_Net(mask_X)
Madv_logits, Madv_probs, Madv_end_point = bulid_Net(mask_X_adv)

with tf.name_scope("attation_loss"):
    a_1 = -tf.reduce_mean(tf.square(rar_probs - Mrar_probs))
    a_2 = -tf.reduce_mean(tf.square(adv_probs - Madv_probs))
    # a_1 = -slim.losses.softmax_cross_entropy(Mrar_logits, Y)
    # a_2 = -slim.losses.softmax_cross_entropy(Madv_logits, Y)
    attention_loss = a_1 + a_2

with tf.name_scope("total_loss"):
    total_loss = a * classification_loss + b * attention_loss + c * grad_cam_loss

with tf.name_scope("loss"):
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('classification_loss', classification_loss)
    tf.summary.scalar('attention_loss', attention_loss)
    tf.summary.scalar('grad_cam_loss', grad_cam_loss)

correct_p = tf.equal(tf.argmax(rar_probs, 1), (tf.argmax(Y, 1)))
accuracy = tf.reduce_mean(tf.cast(correct_p, "float"))
tf.summary.scalar('accuracy', accuracy)
train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)
summary_op = tf.summary.merge_all()


def read_and_decode(filename, batch_size):
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

    image = tf.reshape(A, shape=[32, 32, 3])

    image = tf.image.random_flip_left_right(image)

    preprocessed = tf.multiply(tf.subtract(image / 255, 0.5), 2.0)

    img_batch = tf.train.shuffle_batch([preprocessed, Label], batch_size=batch_size, num_threads=64, capacity=10000,
                                       min_after_dequeue=5000)
    return img_batch


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
train_batch = read_and_decode(filename, _BATCH_SIZE)
test_batch = test_read_and_decode(test_filename, _BATCH_SIZE)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

train_writer = tf.summary.FileWriter("model/log/train", sess.graph)
adv_writer = tf.summary.FileWriter("model/log/adv", sess.graph)
result_file = 'equal_iou' + str(key) + str(a) + '_' + str(b) + '_' + str(c) + '.txt'
# sess.graph.finalize()
import matplotlib.pyplot as plt


def show_img(img, rar, adv):
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    rar = np.reshape(rar, (8, 8))
    plt.imshow(rar)
    plt.subplot(1, 3, 3)
    adv = np.reshape(adv, (8, 8))
    plt.imshow(adv)

    plt.show()


if os.path.isfile(result_file):
    os.remove(result_file)
for i in range(50000):
    step = 0
    batch = sess.run(train_batch)
    ious = np.ones(_BATCH_SIZE)
    adv_sample = sess.run(fixed_adv_sample_get_op, feed_dict={X: batch[0], Y: batch[1], keep_prop: 1.0, IOUS: ious})
    # show_rar,show_adv=sess.run(show_rar_advs, feed_dict={X: batch[0], X_adv: adv_sample, Y: batch[1]})
    # show_img(batch[0][0],show_rar,show_adv)
    _, acc, summery = sess.run([train_op, accuracy, summary_op],
                               feed_dict={X: batch[0], X_adv: adv_sample, Y: batch[1], keep_prop: 0.5})

    while True:
        if step == 3:
            break
        ious = sess.run(rar_adv_ious, feed_dict={X: batch[0], X_adv: adv_sample, Y: batch[1], keep_prop: 1.0})
        adv_sample = sess.run(fixed_adv_sample_get_op,
                              feed_dict={X: adv_sample, Y: batch[1], keep_prop: 1.0, IOUS: ious})
        _, acc, summery = sess.run([train_op, accuracy, summary_op],
                                   feed_dict={X: batch[0], X_adv: adv_sample, Y: batch[1], keep_prop: 0.5})
        step += 1
        print(i,step,ious[0])


    if 1 % 100 == 0:

        testbatch = sess.run(test_batch)
        test_acc, test_summ = sess.run([accuracy, summary_op], feed_dict={X: testbatch[0], Y: batch[1], keep_prop: 1.0})
        train_writer.add_summary(summery, i)
        print(acc, i)
    if i % 1000 == 0 and i != 0:
        f_w = open(result_file, 'a')
        f_w.write(str(acc) + " " + str(i) + "\n")
        f_w.close()
        save()

save()

ious = np.ones(_BATCH_SIZE)

batch = sess.run(train_batch)
R_cam = sess.run(rar_grad_cam, feed_dict={X: batch[0], Y: batch[1], keep_prop: 1.0})
R_mask = sess.run(mask_X, feed_dict={X: batch[0], Y: batch[1], keep_prop: 1.0})
R_mask = np.nan_to_num(R_mask)
ADV_s = sess.run(fixed_adv_sample_get_op, feed_dict={X: batch[0], keep_prop: 1.0, IOUS: ious})
A_cam = sess.run(rar_grad_cam, feed_dict={X: ADV_s, Y: batch[1], keep_prop: 1.0})
A_mask = sess.run(mask_X, feed_dict={X: ADV_s, Y: batch[1], keep_prop: 1.0})
A_mask = np.nan_to_num(A_mask)
time_end = time.time()

print("进行RAR测试集测试:")
ACC = 0
adv_ACC = 0
noise_adv_ACC = 0
double_adv_ACC = 0
for i in range(10000 // _BATCH_SIZE):
    testbatch = sess.run(test_batch)
    adv_sample = sess.run(fixed_adv_sample_get_op, feed_dict={X: testbatch[0], keep_prop: 1.0, IOUS: ious})
    noise_adv_sample = sess.run(NOISE_adv_sample_get_op, feed_dict={X: testbatch[0], keep_prop: 1.0, IOUS: ious})
    double_adv_sample = sess.run(fixed_adv_sample_get_op, feed_dict={X: adv_sample, keep_prop: 1.0, IOUS: ious})
    noise_adv_ACC += sess.run(accuracy, feed_dict={X: noise_adv_sample, Y: testbatch[1], keep_prop: 1.0})
    ACC += sess.run(accuracy, feed_dict={X: testbatch[0], Y: testbatch[1], keep_prop: 1.0})
    adv_ACC += sess.run(accuracy, feed_dict={X: adv_sample, Y: testbatch[1], keep_prop: 1.0})
    double_adv_ACC += sess.run(accuracy, feed_dict={X: double_adv_sample, Y: testbatch[1], keep_prop: 1.0})
print(ACC / (10000 // _BATCH_SIZE))
print(adv_ACC / (10000 // _BATCH_SIZE))
print(noise_adv_ACC / (10000 // _BATCH_SIZE))
print(double_adv_ACC / (10000 // _BATCH_SIZE))
f_w = open('result.txt', 'a')
f_w.write(str(key) + "_" + "equal_iou " + " ")
f_w.write(str(ACC / (10000 // _BATCH_SIZE)) + " " + str(adv_ACC / (10000 // _BATCH_SIZE)) + " ")
f_w.write(str(noise_adv_ACC / (10000 // _BATCH_SIZE)) + " " + str(double_adv_ACC / (10000 // _BATCH_SIZE)) + "\n")
f_w.close()

f_w = open(result_file, 'a')

f_w.write(str(ACC / (10000 // _BATCH_SIZE)) + " " + str(adv_ACC / (10000 // _BATCH_SIZE)) + " ")
f_w.write(str(noise_adv_ACC / (10000 // _BATCH_SIZE)) + " " + str(double_adv_ACC / (10000 // _BATCH_SIZE)) + "\n")
f_w.close()
