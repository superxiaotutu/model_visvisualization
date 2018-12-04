import tensorflow as tf
import tensorflow.contrib.slim as slim
import cifarnet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
_BATCH_SIZE = 1
lr = 0.0001

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
    return resize_cam, cam


def mask_image(img, grad_cam):
    img = tf.reshape(img, [-1, 32, 32, 3])
    cam_clip = tf.sign(tf.nn.relu(grad_cam - tf.reduce_mean(grad_cam)))
    reverse_cam = tf.abs(cam_clip - 1)
    return reverse_cam * img

# model_clean_cifar
# model_cifar_mard
# model8.0_4.0_1.0
model_name='model8.0_4.5_1.0'
def restore():
    saver = tf.train.Saver()
    saver.restore(sess, model_name+"/model.ckpt")


adv_sample_get_op = stepll_adversarial_images(X, tf.random_uniform([1], 0, 0.3))
fixed_adv_sample_get_op = stepll_adversarial_images(X, 0.15)
NOISE_adv_sample_get_op = stepllnoise_adversarial_images(X, 0.15)

rar_logits, rar_probs, rar_end_point = bulid_Net(X)
adv_logits, adv_probs, adv_end_point = bulid_Net(X_adv)


rar_grad_cam, rar_rarcam = grad_cam(rar_end_point, Y)
adv_grad_cam, adv_rarcam = grad_cam(adv_end_point, Y)


mask_X = mask_image(X, rar_grad_cam)
mask_X_adv = mask_image(X_adv, adv_grad_cam)

Mrar_logits, Mrar_probs, Mrar_end_point = bulid_Net(mask_X)
Madv_logits, Madv_probs, Madv_end_point = bulid_Net(mask_X_adv)


correct_p = tf.equal(tf.argmax(rar_probs, 1), (tf.argmax(Y, 1)))
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
train_filename = 'cifar_data/cifar10_train.tfrecord'

test_batch = test_read_and_decode(test_filename, _BATCH_SIZE)
coord = tf.train.Coordinator()
train_batch=test_read_and_decode(train_filename,_BATCH_SIZE)
threads = tf.train.start_queue_runners(coord=coord, sess=sess)


restore()
sess.graph.finalize()
import numpy as np

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
for i in range(10000 // _BATCH_SIZE):
    print(i)
    testbatch = sess.run(test_batch)
    adv_sample = sess.run(fixed_adv_sample_get_op, feed_dict={X: testbatch[0], keep_prop: 1.0})
    noise_adv_sample = sess.run(NOISE_adv_sample_get_op, feed_dict={X: testbatch[0], keep_prop: 1.0})
    double_adv_sample = sess.run(fixed_adv_sample_get_op, feed_dict={X: adv_sample, keep_prop: 1.0})
    arr1.append(adv_sample[0])
    arr2.append(noise_adv_sample[0])
    arr3.append(double_adv_sample[0])
    arr4.append(testbatch[0][0])
    arr5.append(testbatch[1][0])


np.savez('result_8451_cifar.npz',arr1,arr2,arr3,arr4,arr5)


# with open(model_name+'.txt','a') as f_w:
#     f_w.write(str(ACC / (10000 // _BATCH_SIZE)) + " " + str(adv_ACC / (10000 // _BATCH_SIZE)) + " ")
#     f_w.write(str(noise_adv_ACC / (10000 // _BATCH_SIZE)) + " " + str(double_adv_ACC / (10000 // _BATCH_SIZE)) + "\n")
# PGD_v = tf.Variable(tf.zeros([_BATCH_SIZE, 32, 32, 3]))
# assign_op = tf.assign(PGD_v, X)
# logits, probs, end_point = bulid_Net(PGD_v)
# # 自动更新
# loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
# train_op = tf.train.GradientDescentOptimizer(1).minimize(-loss, var_list=[PGD_v])
# # project
# epsilon = 20 / 255
# below = X - epsilon
# above = X + epsilon
# projected = tf.clip_by_value(tf.clip_by_value(PGD_v, below, above), -1, 1)
# with tf.control_dependencies([projected]):
#     project_step = tf.assign(PGD_v, projected)
# PGD_ACC = 0
#
# for i in range(10000 // _BATCH_SIZE):
#     testbatch = sess.run(test_batch)
#     sess.run(assign_op, feed_dict={X: testbatch[0]})
#     for step in range(100):
#         sess.run(train_op, feed_dict={Y: testbatch[1], keep_prop: 1.0})
#         sess.run(project_step, feed_dict={X: testbatch[0]})
#     PGD_sample = sess.run(PGD_v)
#     PGD_ACC += sess.run(accuracy, feed_dict={X: PGD_sample, Y: testbatch[1], keep_prop: 1.0})
# print(PGD_ACC / (10000 // _BATCH_SIZE))
