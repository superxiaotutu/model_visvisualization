import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import matplotlib.pyplot as plt
import PIL
import numpy as np
import xml
from PIL import Image

sess = tf.InteractiveSession()

height, width = 299, 299
X = tf.placeholder(tf.float32, [None, height, width, 3])
Y = tf.placeholder(tf.float32, [None, 80])

def inception(image, reuse = tf.AUTO_REUSE):
    preprocessed = tf.multiply(tf.subtract(image/255, 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_point = nets.inception.inception_v3(preprocessed, 80, is_training=True, reuse=reuse)
        probs = tf.nn.softmax(logits) # probabilities
    return logits, probs, end_point

logits, probs, end_point = inception(X)

def read_and_decode(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    feature = {'label': tf.FixedLenFeature([], tf.int64), 'test_img': tf.FixedLenFeature([], tf.string)}
    features = tf.parse_single_example(serialized_example, features=feature)

    Label = tf.one_hot(features['label'], 80)
    A = tf.decode_raw(features['test_img'], tf.uint8)

    A = tf.reshape(A, shape=[299, 299, 3])

    Label = tf.expand_dims(Label, 0)
    A = tf.expand_dims(A, 0)
    # img_batch = tf.train.batch_join([Label, A], batch_size=batch_size)
    return [Label, A]

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(probs, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

filename = '/run/media/kirin/DOC/ImageNet2012/test.tfrecords'
test_batch = read_and_decode(filename, 1)

summary_op = tf.summary.merge_all()
# train_writer = tf.summary.FileWriter("log/", sess.graph)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
i = 0
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, "model/model_all.ckpt")

total_num = 17120
acc_num = 0

for iter_num in range(total_num):
    one_of_train_batch = sess.run(test_batch)

    # print(one_of_train_batch[0])
    # plt.imshow(np.reshape((one_of_train_batch[1]+1)/2, [299, 299, 3]))
    # plt.show()

    acc_num += sess.run([accuracy], feed_dict={X: one_of_train_batch[1], Y: one_of_train_batch[0]})[0]
    print('Iter: {}, ACC: {}'.format(iter_num, acc_num))

print('Test_ACC: {}'.format(acc_num/total_num))
