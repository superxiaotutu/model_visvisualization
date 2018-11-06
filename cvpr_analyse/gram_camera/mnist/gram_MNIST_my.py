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

_BATCH_SIZE = 50

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
    # resize_cam = resize_cam / tf.reduce_max(resize_cam)
    return resize_cam


def mask_image(img, grad_cam):
    img = tf.reshape(img, [-1, 28, 28, 1])
    cam_clip = tf.sign(tf.nn.relu(grad_cam - tf.reduce_mean(grad_cam)))
    reverse_cam = tf.abs(cam_clip - 1)
    return reverse_cam * img


adv_sample_get_op = stepll_adversarial_images(X, tf.random_uniform([1], 0, 0.3))
fixed_adv_sample_get_op = stepll_adversarial_images(X, 0.15)


rar_logits, rar_probs, rar_end_point = bulid_Net(X)
adv_logits, adv_probs, adv_end_point = bulid_Net(X_adv)

with tf.name_scope("classifition_loss"):
    loss_1 = slim.losses.softmax_cross_entropy(rar_logits, Y)
    loss_2 = slim.losses.softmax_cross_entropy(adv_logits, Y)
    classification_loss = loss_1 + loss_2

rar_grad_cam = grad_cam(rar_end_point, Y)
adv_grad_cam = grad_cam(adv_end_point, Y)

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
    total_loss = 1*classification_loss + 0.5*attention_loss + 1*grad_cam_loss

correct_p = tf.equal(tf.argmax(rar_probs, 1), (tf.argmax(Y, 1)))
accuracy = tf.reduce_mean(tf.cast(correct_p, "float"))
train_op = tf.train.GradientDescentOptimizer(0.005).minimize(total_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
result_file="mnist_my_1_0.5_1.txt"
def save():
    saver = tf.train.Saver()
    saver.save(sess, "model1_0.5_1_my/model.ckpt")


def restore():
    saver = tf.train.Saver()
    saver.restore(sess, "model1_0.5_1_my/model.ckpt")
for i in range(24000):
    batch = mnist.train.next_batch(50)
    adv_sample = sess.run(fixed_adv_sample_get_op, feed_dict={X: batch[0], keep_prop: 1.0})
    _, acc = sess.run([train_op, accuracy], feed_dict={X: batch[0], X_adv: adv_sample, Y: batch[1], keep_prop: 0.5})
    adv_sample_N1 = sess.run(fixed_adv_sample_get_op, feed_dict={X: adv_sample, keep_prop: 1.0})
    _, acc = sess.run([train_op, accuracy], feed_dict={X: batch[0], X_adv: adv_sample_N1, Y: batch[1], keep_prop: 0.5})

    if i % 100 == 0:
        print(acc)
    if i% 4000 ==0:
        with open(result_file,'a')as f:
            f.write(str(acc)+"\n")
    save()
save()
batch = mnist.train.next_batch(50)
R_cam = sess.run(rar_grad_cam, feed_dict={X: batch[0], Y: batch[1], keep_prop: 1.0})
R_mask = sess.run(mask_X, feed_dict={X: batch[0], Y: batch[1], keep_prop: 1.0})
R_mask = np.nan_to_num(R_mask)
ADV_s = sess.run(fixed_adv_sample_get_op, feed_dict={X: batch[0], keep_prop: 1.0})
A_cam = sess.run(rar_grad_cam, feed_dict={X: ADV_s, Y: batch[1], keep_prop: 1.0})
A_mask = sess.run(mask_X, feed_dict={X: ADV_s, Y: batch[1], keep_prop: 1.0})
A_mask = np.nan_to_num(A_mask)



print("进行RAR测试集测试:")
ACC = 0
adv_ACC = 0
for i in range(200):
    testbatch = mnist.test.next_batch(50)
    adv_sample = sess.run(fixed_adv_sample_get_op, feed_dict={X: testbatch[0], keep_prop: 1.0})
    ACC += sess.run(accuracy, feed_dict={X: testbatch[0], Y: testbatch[1], keep_prop: 1.0})
    adv_ACC += sess.run(accuracy, feed_dict={X: adv_sample, Y: testbatch[1], keep_prop: 1.0})
print(ACC / 200)
print(adv_ACC / 200)
with open(result_file, 'a')as f:
    f.write(str(ACC/200) + "\n")
    f.write(str(adv_ACC/200) + "\n")
