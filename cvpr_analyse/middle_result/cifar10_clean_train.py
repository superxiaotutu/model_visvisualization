import time

import tensorflow as tf
import tensorflow.contrib.slim as slim
import cifarnet
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"


_BATCH_SIZE = 80
lr = 0.0001

X = tf.placeholder(tf.float32, [_BATCH_SIZE, 32, 32, 3])
Y = tf.placeholder(tf.float32, [_BATCH_SIZE, 10])
keep_prop = tf.placeholder("float")


def bulid_Net(image, reuse=tf.AUTO_REUSE):
    image = tf.reshape(image, [-1, 32, 32, 3])
    with tf.variable_scope(name_or_scope='CifarNet', reuse=reuse):
        arg_scope = cifarnet.cifarnet_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_point = cifarnet.cifarnet(image, 10, is_training=True, dropout_keep_prob=keep_prop)
            probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point







rar_logits, rar_probs, rar_end_point = bulid_Net(X)

with tf.name_scope("classifition_loss"):
    loss_1 = slim.losses.softmax_cross_entropy(rar_logits, Y)
    regularization = tf.add_n(slim.losses.get_regularization_losses())
    classification_loss = loss_1  + regularization



with tf.name_scope("total_loss"):
    total_loss = 8 * classification_loss

with tf.name_scope("loss"):
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('classification_loss', classification_loss)


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


def save(num):
    saver = tf.train.Saver()
    saver.save(sess, "model_clean_"+str(num)+ "/model.ckpt")
filename = 'cifar_data/cifar10_train.tfrecord'
test_filename = 'cifar_data/cifar10_test.tfrecord'
train_batch = read_and_decode(filename, _BATCH_SIZE)
test_batch = test_read_and_decode(test_filename, _BATCH_SIZE)
coord = tf.train.Coordinator()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

threads = tf.train.start_queue_runners(coord=coord, sess=sess)

train_writer = tf.summary.FileWriter("model/log/train", sess.graph)
# test_writer = tf.summary.FileWriter("model/log/test", sess.graph)
adv_writer = tf.summary.FileWriter("model/log/adv", sess.graph)

result_file = 'result_clean.txt'
f_w = open(result_file, 'a')
f_w.write('ini')
f_w.close()
for i in range(50000):
    batch = sess.run(train_batch)
    _, acc, summery = sess.run([train_op, accuracy, summary_op],
                               feed_dict={X: batch[0], Y: batch[1], keep_prop: 0.5})

    if 1 % 100 == 0:
        testbatch = sess.run(test_batch)
        test_acc, test_summ = sess.run([accuracy, summary_op], feed_dict={X: testbatch[0], Y: batch[1], keep_prop: 1.0})
    if 1 % 1000 == 0:
        f_w = open(result_file, 'a')
        f_w.write(str(acc) + " " + str(i) + "\n")
        f_w.close()
    if i % 10000 == 0:
        f_w = open(result_file, 'a')
        f_w.write(str(acc) + " " + str(i) + "\n")
        f_w.close()
        save(i)
save('final')
