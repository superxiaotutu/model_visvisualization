import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import numpy as np
import os
import PIL
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
plt.switch_backend('agg')

_BATCH_SIZE = 1
X = tf.placeholder(tf.float32, [_BATCH_SIZE, 299, 299, 3])
Y = tf.placeholder(tf.int32, [_BATCH_SIZE])
LABEL = tf.placeholder(tf.int32)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)


def inception(image, reuse=tf.AUTO_REUSE):
    preprocessed = tf.multiply(tf.subtract(image, 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, end_point = nets.inception.inception_v3(preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point




one_hot_ll_class = tf.one_hot(Y, 1000)


rar_logits, rar_probs, rar_end_point = inception(X)

is_corrects = tf.equal(tf.argmax(rar_probs, 1), (tf.argmax(one_hot_ll_class, 1)))



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
            ground_truth[matrix[1]:matrix[3], matrix[0]:matrix[2]] = 1
    return ground_truth



def get_rar_adv_iou(rar, adv):
    rar_count = rar[rar == 1].size
    adv_count = adv[adv == 1].size
    sum = rar + adv
    IOU = sum[sum == 2].size / sum[sum != 0].size
    return rar_count, adv_count, IOU


from skimage.transform import resize


def make_cam(x):
    threshold = np.sort(np.reshape(x, (64)))[-int(key)]
    x = x - threshold
    x = resize(x, [299, 299])
    x[x < 0] = 0
    x[x > 0] = 1
    return x


if __name__ == '__main__':
    loop_num = 0
    IOU_sum = 0
    labels_file = 'imagenet_labels.txt'
    results_file = str(key) + 'rar_groundtruth_iou' + '.txt'

    if os.path.exists(results_file):
        os.remove(results_file)
    defense_iou = 0
    defense_count = 0
    attack_iou = 0
    attack_count = 0
    rar_ground_iou_sum = 0
    adv_ground_iou_sum = 0

    label_paths = []
    with open(labels_file, 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            imgs = []
            labels = []
            label_letter = line.split(' ')
            ground_truths = []
            label_letter = label_letter[0]
            img_class = index
            dir_name = 'img_val/' + str(label_letter)
            for root, dirs, files in os.walk(dir_name):
                for file in files:
                    img_path = dir_name + '/' + file
                    label_path = 'val/' + str(file)[:-4] + 'xml'
                    imgs.append(load_img(img_path))
                    labels.append(index)
                    ground_truths.append(get_gound_truth(label_path))
                    label_paths.append(img_path)

            is_defenses = sess.run(is_defense, feed_dict={X: imgs})
            rar_ps = sess.run(rar_probs, feed_dict={X: imgs})
            adv_ps = sess.run(adv_probs, feed_dict={X: imgs})
