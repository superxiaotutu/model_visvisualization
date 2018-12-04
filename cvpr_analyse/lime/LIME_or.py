import tensorflow as tf

slim = tf.contrib.slim
import sys

sys.path.append('slim')

import matplotlib.pyplot as plt
import numpy as np
from nets import inception
from preprocessing import inception_preprocessing
import os
from lime import lime_image
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
plt.switch_backend('agg')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

image_size = inception.inception_v3.default_image_size


def transform_img_fn(path_list):
    out = []
    for f in path_list:
        image_raw = tf.image.decode_jpeg(open(f, 'rb').read(), channels=3)
        image = inception_preprocessing.preprocess_image(image_raw, image_size, image_size, is_training=False)
        out.append(image)
    return session.run([out])[0]


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
processed_images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))

with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, _ = inception.inception_v3(processed_images, num_classes=1001, is_training=False)
probabilities = tf.nn.softmax(logits)

checkpoints_dir = './'
init_fn = slim.assign_from_checkpoint_fn(
    os.path.join(checkpoints_dir, 'inception_v3.ckpt'),
    slim.get_model_variables('InceptionV3'))
init_fn(session)


def step_target_class_adversarial_images(x, eps, one_hot_target_class):
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                    logits,
                                                    label_smoothing=0.1,
                                                    weights=1.0)
    x_adv = x - eps * tf.sign(tf.gradients(cross_entropy, x)[0])
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
    return tf.stop_gradient(x_adv)


def stepll_adversarial_images(x, eps):
    least_likely_class = tf.argmin(logits, 1)
    one_hot_ll_class = tf.one_hot(least_likely_class, 1000)
    return step_target_class_adversarial_images(x, eps, one_hot_ll_class)


def stepllnoise_adversarial_images(x, eps):
    least_likely_class = tf.argmin(logits, 1)
    one_hot_ll_class = tf.one_hot(least_likely_class, 10)
    x_noise = x + eps / 2 * tf.sign(tf.random_normal(x.shape))
    return step_target_class_adversarial_images(x_noise, eps / 2, one_hot_ll_class)


def predict_fn(images):
    return session.run(probabilities, feed_dict={processed_images: images})


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


tmp = time.time()
from skimage.segmentation import mark_boundaries

if __name__ == '__main__':
    labels_file = 'imagenet_labels.txt'
    results_file = 'result_lime.txt'
    if os.path.isfile (results_file):
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
        for label_index, line in enumerate(lines):
            imgs = []
            labels = []
            label_letter = line.split(' ')
            ground_truths = []
            label_letter = label_letter[0]
            img_class = label_index
            dir_name = 'img_val/' + str(label_letter)
            for root, dirs, files in os.walk(dir_name):
                for file in files:
                    img_path = dir_name + '/' + file
                    label_path = 'val/' + str(file)[:-4] + 'xml'
                    imgs.append(img_path)
                    labels.append(label_index)
                    ground_truths.append(get_gound_truth(label_path))
                    label_paths.append(img_path)
            images = transform_img_fn(imgs)
            preds = predict_fn(images)
            with open(results_file, 'a', encoding='utf-8') as f_w:
                for j in range(50):
                    label_pred = np.argmax(preds[j])
                    print(label_pred, labels[j])
                    if label_pred == labels[j]:
                        explainer = lime_image.LimeImageExplainer()
                        explanation = explainer.explain_instance(images[j], predict_fn, top_labels=1, hide_color=0,
                                                                 num_samples=1000)
                        temp, mask = explanation.get_image_and_mask(label_pred, positive_only=False, num_features=5,
                                                                    hide_rest=False)
                        print(mask.shape)
                        plt.figure()
                        plt.subplot(1, 3, 1)
                        temp = np.clip(temp, 0, 1)
                        plt.imshow(temp)
                        plt.subplot(1, 3, 2)
                        mask = np.clip(mask, 0, 1)
                        plt.imshow(mask)
                        plt.subplot(1, 3, 3)
                        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

                        plt.savefig(str(label_index) + "_" + str(j) + '.png')
print(time.time() - tmp)
