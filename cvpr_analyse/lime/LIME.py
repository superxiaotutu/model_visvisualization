import os
import cv2

import tensorflow as tf

slim = tf.contrib.slim
import sys

sys.path.append('slim')
import matplotlib.pyplot as plt
import numpy as np
from nets import inception
import tensorflow.contrib.slim.nets as nets
from preprocessing import inception_preprocessing
from lime import lime_image
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

plt.switch_backend('agg')

image_size = inception.inception_v3.default_image_size


def transform_img_fn(path_list):
    out = []
    for f in path_list:
        image_raw = tf.image.decode_jpeg(open(f, 'rb').read(), channels=3)
        image = inception_preprocessing.preprocess_image(image_raw, image_size, image_size, is_training=False)
        out.append(image)
    return sess.run([out])[0]


def get_names():
    filename = 'imagenet_lsvrc_2015_synsets.txt'
    synset_list = [s.strip() for s in open(filename).readlines()]
    filename = 'imagenet_metadata.txt'
    synset_to_human_list = open(filename).readlines()
    synset_to_human = {}
    for s in synset_to_human_list:
        parts = s.strip().split('\t')
        assert len(parts) == 2
        synset = parts[0]
        human = parts[1]
        synset_to_human[synset] = human

    label_index = 1
    labels_to_names = {0: 'background'}
    for synset in synset_list:
        name = synset_to_human[synset]
        labels_to_names[label_index] = name
        label_index += 1

    return labels_to_names


names = get_names()
X = tf.placeholder(tf.float32, [None, 299, 299, 3])
Y = tf.placeholder(tf.int32, [None])


def inceptionv3(image, reuse=tf.AUTO_REUSE):
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, end_point = nets.inception.inception_v3(image, 1001, is_training=False, reuse=reuse)
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


def predict_rar(images):
    return sess.run(rar_probs, feed_dict={X: images})


def predict_adv(images):
    return sess.run(adv_probs, feed_dict={X: images})


def step_target_class_adversarial_images(x, eps, one_hot_target_class):
    logits, prob, end_points = inceptionv3(x)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                    logits,
                                                    label_smoothing=0.1,
                                                    weights=1.0)
    x_adv = x - eps * tf.sign(tf.gradients(cross_entropy, x)[0])
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
    return tf.stop_gradient(x_adv)


def stepll_adversarial_images(x, eps):
    logits, prob, end_points = inceptionv3(x)
    least_likely_class = tf.argmin(logits, 1)
    one_hot_ll_class = tf.one_hot(least_likely_class, 1000)
    return step_target_class_adversarial_images(x, eps, one_hot_ll_class)


def stepllnoise_adversarial_images(x, eps):
    logits, prob, end_points = inceptionv3(x)
    least_likely_class = tf.argmin(logits, 1)
    one_hot_ll_class = tf.one_hot(least_likely_class, 10)
    x_noise = x + eps / 2 * tf.sign(tf.random_normal(x.shape))
    return step_target_class_adversarial_images(x_noise, eps / 2, one_hot_ll_class)


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


fixed_adv_sample_get_op = stepll_adversarial_images(X, 0.15)

rar_logits, rar_probs, rar_end_point = inceptionv3(X)
adv_logits, adv_probs, adv_end_point = inceptionv3(fixed_adv_sample_get_op)

is_defense = tf.equal(tf.argmax(rar_probs, 1), (tf.argmax(adv_probs, 1)))
#
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "inception_v3.ckpt")

tmp = time.time()

def show_img(img, rar_mask, adv_mask, rar_label, adv_label, j):
    plt.figure()

    rar_mask[rar_mask != 2] = 0
    rar_mask = cv2.applyColorMap(np.uint8(255 * rar_mask), cv2.COLORMAP_JET)
    rar_mask = cv2.cvtColor(rar_mask, cv2.COLOR_BGR2RGB)
    img = img.astype(float)
    img /= img.max()
    alpha = 2
    rar_img = img + alpha * rar_mask

    adv_mask[adv_mask != 2] = 0
    adv_mask = cv2.applyColorMap(np.uint8(255 * adv_mask), cv2.COLORMAP_JET)
    adv_mask = cv2.cvtColor(adv_mask, cv2.COLOR_BGR2RGB)
    adv_img = img + alpha * adv_mask

    plt.subplot(1, 5, 1)
    plt.imshow(img)
    plt.subplot(1, 5, 2)
    plt.imshow(rar_mask)

    # plt.imshow(mark_boundaries(images[j] / 2 + 0.5, rar_mask))
    # np.save('a.npy', mark_boundaries(images[j] / 2 + 0.5, rar_mask))
    plt.subplot(1, 5, 3)
    plt.imshow(adv_img)

    # plt.imshow(mark_boundaries(images[j] / 2 + 0.5, adv_mask))
    plt.subplot(1, 5, 4)
    plt.imshow(rar_img)
    plt.subplot(1, 5, 5)
    plt.imshow(adv_img)
    plt.savefig(
        "image/" + str(rar_label) + "_" + str(adv_label) + "_" + str(num_features) + "_" + str(
            j) + '.png')


if __name__ == '__main__':
    num_features = 10
    offset = 332
    labels_file = 'imagenet_labels.txt'
    npz_dir = 'npz_'+str(offset)+'_'+str(offset+100)+'/'
    results_file = 'result_lime' + str(num_features) + '.txt'
    if os.path.isfile(results_file):
        os.remove(results_file)
    if not os.path.isdir(npz_dir):
        os.makedirs(npz_dir)
    defense_iou = 0
    defense_count = 0
    attack_iou = 0
    attack_count = 0
    rar_ground_iou_sum = 0
    adv_ground_iou_sum = 0
    label_paths = []
    with open(labels_file, 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for label_index, line in enumerate(lines[offset:offset+100]):
            imgs = []
            true_labels = []
            label_letter = line.split(' ')
            ground_truths = []
            label_letter = label_letter[0]
            label_index += offset
            dir_name = 'img_val/' + str(label_letter)
            for root, dirs, files in os.walk(dir_name):
                for j,file in enumerate(files):
                    img_path = dir_name + '/' + file
                    label_path = 'val/' + str(file)[:-4] + 'xml'
                    imgs.append(img_path)
                    true_labels.append(label_index)
                    ground_truths.append(get_gound_truth(label_path))
                    label_paths.append(img_path)
                    images = transform_img_fn(imgs)
                    adv_imgs = sess.run(fixed_adv_sample_get_op, feed_dict={X: images})
                    with open(results_file, 'a', encoding='utf-8') as f_w:
                        try:
                            explainer = lime_image.LimeImageExplainer()
                            explanation = explainer.explain_instance(images[j], predict_rar, top_labels=1, hide_color=0,
                                                                     num_samples=1000)
                            for key in (explanation.local_exp.keys()):
                                print(key)
                                rar_label = key
                                break
                            if rar_label == true_labels[j]:

                                rar_temp, rar_mask = explanation.get_image_and_mask(rar_label, positive_only=False,
                                                                                    num_features=num_features,
                                                                                    hide_rest=False)

                                explanation = explainer.explain_instance(adv_imgs[j], predict_adv, top_labels=1,
                                                                         hide_color=0,
                                                                         num_samples=1000)
                                for key in (explanation.local_exp.keys()):
                                    print(key)
                                    adv_label = key
                                    break
                                adv_temp, adv_mask = explanation.get_image_and_mask(adv_label, positive_only=False,
                                                                                    num_features=num_features,
                                                                                    hide_rest=False)

                                np.savez(
                                    npz_dir + str(rar_label) + "_" + str(adv_label) + "_" + str(num_features) + "_" + str(
                                        j), images[j], rar_mask, adv_mask, ground_truths[j])

                        except Exception as e:
                            print(e)
                            with open(results_file, 'a')as f:
                                f.write(e)
                                f.write(str(label_paths[j]))
