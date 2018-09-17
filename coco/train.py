import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import matplotlib.pyplot as plt
import PIL
import numpy as np

sess = tf.InteractiveSession()

height, width = 299, 299
X = tf.placeholder(tf.float32, [None, height, width, 3])
C = tf.placeholder(tf.float32, [None, height, width, 3])
Y = tf.placeholder(tf.float32, [None, 80])

def save(N = None):
    saver = tf.train.Saver()
    if N != None:
        saver.save(sess, "model/model_all_"+ str(N) +".ckpt", global_step= N)
    else:
        saver.save(sess, "model/model_all.ckpt")

def inception(image, reuse = tf.AUTO_REUSE):
    preprocessed = tf.multiply(tf.subtract(image/255, 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_point = nets.inception.inception_v3(preprocessed, 80, is_training=True, reuse=reuse)
        probs = tf.nn.softmax(logits) # probabilities
    return logits, probs, end_point

logits, probs, end_point = inception(X)
logits_c, probs_c, end_point_c = inception(C)

# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits), name='loss')
loss_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)) - tf.reduce_mean(tf.square(Y-probs_c))

with tf.name_scope('loss'):
    summary_loss = tf.summary.scalar('Loss', loss_c)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(probs, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        summary_acc = tf.summary.scalar('accuracy', accuracy)

train = tf.train.AdamOptimizer(0.00001).minimize(loss_c)

def read_and_decode(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    feature = {'label': tf.FixedLenFeature([], tf.int64), 'A_img': tf.FixedLenFeature([], tf.string), 'B_img': tf.FixedLenFeature([], tf.string), 'C_img': tf.FixedLenFeature([], tf.string)}
    features = tf.parse_single_example(serialized_example, features=feature)

    Label = tf.one_hot(features['label'], 80)
    A = tf.decode_raw(features['A_img'], tf.uint8)
    B = tf.decode_raw(features['B_img'], tf.uint8)
    C = tf.decode_raw(features['C_img'], tf.uint8)

    A = tf.reshape(A, shape=[299, 299, 3])
    B = tf.reshape(B, shape=[299, 299, 3])
    C = tf.reshape(C, shape=[299, 299, 3])

    img_batch = tf.train.shuffle_batch([Label, A, B, C], batch_size=batch_size, num_threads=64, capacity=5000, min_after_dequeue=2500)
    return img_batch

def test_read_and_decode(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    feature = {'label': tf.FixedLenFeature([], tf.int64), 'test_img': tf.FixedLenFeature([], tf.string)}
    features = tf.parse_single_example(serialized_example, features=feature)

    Label = tf.one_hot(features['label'], 80)
    A = tf.decode_raw(features['test_img'], tf.uint8)

    A = tf.reshape(A, shape=[299, 299, 3])

    img_batch = tf.train.shuffle_batch([Label, A], batch_size=batch_size, num_threads=64, capacity=4000, min_after_dequeue=2000)
    return img_batch

filename = '/run/media/kirin/DOC/ImageNet2012/train.tfrecords'
test_filename = '/run/media/kirin/DOC/ImageNet2012/test.tfrecords'
train_batch = read_and_decode(filename, 30)
test_batch = test_read_and_decode(test_filename, 30)

summary_op = tf.summary.merge_all()

train_writer = tf.summary.FileWriter("log/train", sess.graph)
test_writer = tf.summary.FileWriter("log/test", sess.graph)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
i = 0
sess.run(tf.global_variables_initializer())

# saver = tf.train.Saver()
# saver.restore(sess, "model/model_all.ckpt")

while not coord.should_stop():
    one_of_train_batch = sess.run(train_batch)
    _, loss_num, acc_num, summary_str = sess.run([train, loss_c, accuracy, summary_op], feed_dict={X: one_of_train_batch[1], Y: one_of_train_batch[0], C: one_of_train_batch[3]})
    train_writer.add_summary(summary_str, i)
    print('Iter: {}, LOSS: {}, ACC: {}'.format(i, loss_num, acc_num))
    i += 1
    if i % 10 == 0:
        one_of_test_batch = sess.run(test_batch)
        _, test_summary = sess.run([accuracy, summary_acc], feed_dict={X: one_of_test_batch[1], Y: one_of_test_batch[0]})
        test_writer.add_summary(test_summary, i)
    if i % 1000 == 0 and i != 0:
        save()
    if i % 10000 == 0 and i != 0:
        save(i)
    if i == 100000:
        coord.request_stop()
