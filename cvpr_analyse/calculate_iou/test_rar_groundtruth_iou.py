import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import numpy as np
import os
import PIL
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

_BATCH_SIZE = 1
X = tf.placeholder(tf.float32, [_BATCH_SIZE, 299, 299, 3])
Y = tf.placeholder(tf.int32, [_BATCH_SIZE])
LABEL = tf.placeholder(tf.int32)
sess = tf.InteractiveSession()


def inception(image, reuse=tf.AUTO_REUSE):
    preprocessed = tf.multiply(tf.subtract(image, 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, end_point = nets.inception.inception_v3(preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


def step_target_class_adversarial_images(x, eps, one_hot_target_class):
    _, logits, end_points = inception(x)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                    logits,
                                                    label_smoothing=0.1,
                                                    weights=1.0)
    x_adv = x - eps * tf.sign(tf.gradients(cross_entropy, x)[0])
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
    return tf.stop_gradient(x_adv)


def stepll_adversarial_images(x, eps):
    _, logits, end_points = inception(x)
    least_likely_class = tf.argmin(logits, 1)
    one_hot_ll_class = tf.one_hot(least_likely_class, 1000)
    return step_target_class_adversarial_images(x, eps, one_hot_ll_class)


key = 0.8


def grad_cam(end_point, Y, layer_name='Mixed_7c'):
    pre_calss_one_hot = tf.one_hot(Y, depth=1000)
    conv_layer = end_point[layer_name]
    signal = tf.multiply(end_point['Logits'][:, 1:], pre_calss_one_hot)
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
    # resize_cam = tf.image.resize_images(cam, [299, 299])
    '''新加sign'''
    # print(resize_cam.shape)
    # resize_cam = tf.reshape(resize_cam, [299*299])
    # threshold = tf.nn.top_k(resize_cam, k=299*299//6).values[-1]
    # resize_cam=resize_cam-threshold
    # resize_cam = tf.nn.relu(resize_cam)
    return (cam)


fixed_adv_sample_get_op = stepll_adversarial_images(X, 0.15)

rar_logits, rar_probs, rar_end_point = inception(X)
adv_logits, adv_probs, adv_end_point = inception(fixed_adv_sample_get_op)

is_attack = tf.equal(tf.argmax(rar_probs, 1), (tf.argmax(adv_probs, 1)))

rar_grad_cam = grad_cam(rar_end_point, Y)
adv_grad_cam = grad_cam(adv_end_point, Y)

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


def get_iou(rar, adv, ground_truth):
    if ground_truth[ground_truth == 1].size == 0:
        return False, False, False, False, 'False'
    rar_count = rar[rar == 1].size
    adv_count = adv[adv == 1].size
    ground_count = ground_truth[ground_truth == 1].size
    rar_sum = rar + ground_truth
    adv_sum = adv + ground_truth
    rar_IOU = rar_sum[rar_sum == 2].size / ground_count
    adv_IOU = adv_sum[adv_sum == 2].size / ground_count
    return rar_count, adv_count, ground_count, rar_IOU, adv_IOU


# def make_cam(x):
#     threshold = np.sort(np.reshape(x, (299*299)))[299*299-10000]
#     x = x - threshold
#     x[x < 0] = 0
#     x[x > 0] = 1

def make_cam(x):
    threshold = np.sort(np.reshape(x, (64)))[-2]
    x = x - threshold
    x = resize(x, [299, 299])
    x[x < 0] = 0
    x[x > 0] = 1
    return x


if __name__ == '__main__':
    loop_num = 0
    IOU_sum = 0
    imgs = []
    labels_file = 'imagenet_labels.txt'
    for r,d,f in os.walk('img_val/n01440764'):
        for file in f:
            results_file = 'true_rar_groundtruth_iou_'+str(key)+'.txt'
            imgs=[]
            imgs.append(load_img('img_val/n01440764/'+file))
    # imgs.append(load_img('img_val/n01440764/ILSVRC2012_val_00026064.JPEG'))
            label_path = "b.xml"
            gt = get_gound_truth(label_path)
            rar_maps, adv_maps = sess.run([rar_grad_cam, adv_grad_cam], feed_dict={X: imgs, Y: [0]})
            rar_ps = sess.run(rar_probs, feed_dict={X: imgs})

            rar_maps = np.reshape(rar_maps, (_BATCH_SIZE, 8, 8))
            adv_maps = np.reshape(adv_maps, (_BATCH_SIZE, 8, 8))
            from skimage.transform import resize

            plt.figure()
            plt.subplot(1, 4, 1)
            print(imgs[0].shape)
            print(rar_maps[0])
            print(rar_maps[0].shape)
            print(rar_maps[0].shape)
            plt.imshow(imgs[0])
            plt.subplot(1, 4, 2)
            rar_map = (make_cam(rar_maps[0]))
            plt.imshow(rar_map)

            plt.subplot(1, 4, 3)
            adv_map = make_cam(adv_maps[0])

            plt.imshow(adv_map)
            plt.subplot(1, 4, 4)
            print(get_iou(rar_map, adv_map, gt))
            plt.imshow(gt)
            plt.show()
    # with open(results_file, 'a', encoding='utf-8') as f_w:
    #     for j in range(_BATCH_SIZE):
    #         print(np.argmax(rar_ps[j]),labels[j])
    #         if np.argmax(rar_ps[j])==labels[j]:
    #             rar_count, adv_count,gt_count, rar_g_iou, adv_g_iou = get_iou(rar_maps[j], adv_maps[j], ground_truths[j])
    #             if  adv_g_iou=='False' or rar_g_iou==0:
    #                 filename=str(label_paths[j])[20:]+'_rar_0.png'
    #                 print(filename)
    #                 f_w.write(filename+"\n")
    #                 print(filename)
    #                 plt.figure()
    #                 plt.subplot(1, 4, 1)
    #                 plt.imshow(imgs[j])
    #                 plt.subplot(1, 4, 2)
    #                 plt.imshow(rar_maps[j])
    #                 plt.subplot(1, 4, 3)
    #                 plt.imshow(adv_maps[j])
    #                 plt.subplot(1, 4, 4)
    #                 plt.imshow(ground_truths[j])
    #                 plt.savefig(filename)
    #                 continue
    #             elif adv_g_iou==0:
    #                 filename=str(label_paths[j])[20:]+'_adv_0.png'
    #                 f_w.write(filename)
    #                 print(filename)
    #                 plt.figure()
    #                 plt.subplot(1, 4, 1)
    #                 plt.imshow(imgs[j])
    #                 plt.subplot(1, 4, 2)
    #                 plt.imshow(rar_maps[j])
    #                 plt.subplot(1, 4, 3)
    #                 plt.imshow(adv_maps[j])
    #                 plt.subplot(1, 4, 4)
    #                 plt.imshow(ground_truths[j])
    #                 plt.savefig(filename)
    #             loop_num += 1
    #             rar_ground_iou_sum += rar_g_iou
    #             adv_ground_iou_sum += adv_g_iou
    #
    #             print(loop_num)
    #             print(rar_count, adv_count, gt_count,rar_g_iou, adv_g_iou)
    #
    # f_w.write(
    #     str(loop_num) + " " + str(
    #         rar_count)
    #     + ' ' + str(adv_count)+' ' + str(rar_g_iou) + ' ' + str(
    #         adv_g_iou) + '\n')
    # with open(results_file, 'a', encoding='utf-8') as f_w:
    #     f_w.write('rar' + " " + str(rar_ground_iou_sum / loop_num) + "\n")
    #     f_w.write('adv' + " " + str(adv_ground_iou_sum / loop_num) + "\n")
