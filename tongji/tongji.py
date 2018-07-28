import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
import PIL
import numpy as np

sess = tf.InteractiveSession()

image = tf.Variable(tf.zeros((299, 299, 3)))

def inception_v3(inputs, num_classes=1000, is_training=True, min_depth=16, depth_multiplier=1.0, reuse=None, scope='InceptionV3'):
  """
  name:
  conv0             | Conv2d_1a_3x3   1*149*149*32
  conv1             | Conv2d_2a_3x3   1*147*147*32
  conv2             | Conv2d_2b_3x3   1*147*147*64
  pool1             | MaxPool_3a_3x3  1*73*73*64
  conv3             | Conv2d_3b_1x1   1*73*73*80
  conv4             | Conv2d_4a_3x3   1*71*71*192
  pool2             | MaxPool_5a_3x3  1*35*35*192
  mixed_35x35x256a  | Mixed_5b        1*35*35*256
  mixed_35x35x288a  | Mixed_5c        1*35*35*288
  mixed_35x35x288b  | Mixed_5d        1*35*35*288
  mixed_17x17x768a  | Mixed_6a        1*17*17*768
  mixed_17x17x768b  | Mixed_6b        1*17*17*768
  mixed_17x17x768c  | Mixed_6c        1*17*17*768
  mixed_17x17x768d  | Mixed_6d        1*17*17*768
  mixed_17x17x768e  | Mixed_6e        1*17*17*768
  mixed_8x8x1280a   | Mixed_7a        1*8*8*1280
  mixed_8x8x2048a   | Mixed_7b        1*8*8*2048
  mixed_8x8x2048b   | Mixed_7c        1*8*8*2048
  """
  with variable_scope.variable_scope(
      scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
    with arg_scope([layers_lib.batch_norm, layers_lib.dropout], is_training=is_training):
      net, end_points = nets.inception.inception_v3_base(inputs, scope=scope, min_depth=min_depth, depth_multiplier=depth_multiplier)
  return net, end_points

def inception(image, reuse = tf.AUTO_REUSE):
    preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        net, end_points = inception_v3(preprocessed, 1001, is_training=False, reuse=reuse)
    return net, end_points

net, end = inception(image)

restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV3/')
]
saver = tf.train.Saver(restore_vars)
saver.restore(sess, "/home/kirin/PycharmProjects/adv_lucid/model/inception_v3.ckpt")

def get_value(input_img):
    net_value, end_value = sess.run([net, end], feed_dict={image: input_img})
    sum = []
    for v in end_value:
        s = np.sum(end_value[v], 0)
        s = np.sum(s, 0)
        s = np.sum(s, 0)
        sum.append(s)
    return np.asarray(sum)

def cal_diff(rar, adv):
    v_rar = get_value(rar)
    v_adv = get_value(adv)
    return v_adv - v_rar

def load_img(path):
    I = PIL.Image.open(path)
    I = I.resize((299, 299)).crop((0, 0, 299, 299))
    I = (np.asarray(I) / 255.0).astype(np.float32)
    return I

namelist = open("namelist.txt")

sum = 0
count = 0
for i in namelist.readlines():
    i = i.strip()
    rar = load_img("299rar/"+i)
    adv = load_img("299adv/"+i)
    sum += cal_diff(rar, adv)
    count += 1
    print(count)
sum = sum/count

# np.save("sum.npy", sum)
