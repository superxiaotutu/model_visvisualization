import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import lucid.modelzoo.vision_models as model
from lucid.misc.io import show,load
from lucid.misc.io.reading import read
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform
from lucid.misc.gradient_override import gradient_override_map
import cv2

model = model.InceptionV1()
model.load_graphdef()

# def raw_class_spatial_attr(img, layer, label, override=None):
#     """How much did spatial positions at a given layer effect a output class?"""
#
#     # Set up a graph for doing attribution...
#     with tf.Graph().as_default(), tf.Session(), gradient_override_map(override or {}):
#         t_input = tf.placeholder_with_default(img, [None, None, 3])
#         T = render.import_model(model, t_input, t_input)
#
#         # Compute activations
#         acts = T(layer).eval()
#
#         if label is None: return np.zeros(acts.shape[1:-1])
#
#         # Compute gradient
#         score = T("softmax2_pre_activation")[0, labels.index(label)]
#         t_grad = tf.gradients([score], [T(layer)])[0]
#         grad = t_grad.eval({T(layer): acts})
#
#         # Linear approximation of effect of spatial position
#         return np.sum(acts * grad, -1)[0]
#
# def raw_spatial_spatial_attr(img, layer1, layer2, override=None):
#     """Attribution between spatial positions in two different layers."""
#
#     # Set up a graph for doing attribution...
#     with tf.Graph().as_default(), tf.Session(), gradient_override_map(override or {}):
#         t_input = tf.placeholder_with_default(img, [None, None, 3])
#         T = render.import_model(model, t_input, t_input)
#
#         # Compute activations
#         acts1 = T(layer1).eval()
#         acts2 = T(layer2).eval({T(layer1): acts1})
#
#         # Construct gradient tensor
#         # Backprop from spatial position (n_x, n_y) in layer2 to layer1.
#         n_x, n_y = tf.placeholder("int32", []), tf.placeholder("int32", [])
#         layer2_mags = tf.sqrt(tf.reduce_sum(T(layer2) ** 2, -1))[0]
#         score = layer2_mags[n_x, n_y]
#         t_grad = tf.gradients([score], [T(layer1)])[0]
#
#         # Compute attribution backwards from each positin in layer2
#         attrs = []
#         for i in range(acts2.shape[1]):
#             attrs_ = []
#             for j in range(acts2.shape[2]):
#                 grad = t_grad.eval({n_x: i, n_y: j, T(layer1): acts1})
#                 # linear approximation of imapct
#                 attr = np.sum(acts1 * grad, -1)[0]
#                 attrs_.append(attr)
#             attrs.append(attrs_)
#     return np.asarray(attrs)
##
# def orange_blue(a,b,clip=False):
#   if clip:
#     a,b = np.maximum(a,0), np.maximum(b,0)
#   arr = np.stack([a, (a + b)/2., b], -1)
#   arr /= 1e-2 + np.abs(arr).max()/1.5
#   arr += 0.3
#   return arr
#
# img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/dog_cat.png")
#
# attrs = raw_spatial_spatial_attr(img, "mixed4d", "mixed5a", override=None)
# attrs = attrs / attrs.max()

_ = render.render_vis(model,"mixed4a_pre_relu:476")
img = np.reshape(_,[128,128,3])

plt.imshow(img)
plt.show()
