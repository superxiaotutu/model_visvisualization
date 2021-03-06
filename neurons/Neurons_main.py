import cv2
import numpy as np
import tensorflow as tf

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.misc.io.showing as showing
from lucid.misc.channel_reducer import ChannelReducer
import lucid.optvis.param as param
import lucid.optvis.objectives as objectives
import lucid.optvis.render as render
from lucid.misc.io import show, load
from lucid.misc.io.reading import read
from lucid.misc.io.showing import _image_url
from lucid.misc.gradient_override import gradient_override_map

model = models.InceptionV1()
model.load_graphdef()
labels_str = read("map_clsloc.txt", encoding='utf8')
labels = [line[line.find(" "):].strip() for line in labels_str.split("\n")]
labels = [label[label.find(" "):].strip().replace("_", " ") for label in labels]
labels = ["dummy"] + labels


def raw_class_group_attr(img, layer, label, group_vecs, override=None):
    """How much did spatial positions at a given layer effect a output class?"""

    # Set up a graph for doing attribution...
    with tf.Graph().as_default(), tf.Session(), gradient_override_map(override or {}):
        t_input = tf.placeholder_with_default(img, [None, None, 3])
        T = render.import_model(model, t_input, t_input)

        # Compute activations
        acts = T(layer).eval()

        if label is None: return np.zeros(acts.shape[1:-1])

        # Compute gradient
        score = T("softmax2_pre_activation")[0, labels.index(label)]
        t_grad = tf.gradients([score], [T(layer)])[0]
        grad = t_grad.eval({T(layer): acts})

        # Linear approximation of effect of spatial position
        return [np.sum(group_vec * grad) for group_vec in group_vecs]


def neuron_groups(imglist, filenamelist, layer, n_groups=6, attr_classes=None):
    # Compute activations
    filename = ''
    for f in filenamelist:
        filename += f
    with open('result/' + filename+'.html', 'a') as f:
        f.write('''<!DOCTYPE html>
                        <html>
                        <head >
                          <title>%s</title>
                              <script src='GroupWidget_1cb0e0d.js'></script>
                        </head>
                        <body>'''%(filename))
    for key, img in enumerate(imglist):
        if attr_classes is None:
            attr_classes = []
        with tf.Graph().as_default(), tf.Session():
            t_input = tf.placeholder_with_default(img, [None, None, 3])
            T = render.import_model(model, t_input, t_input)
            acts = T(layer).eval()

        # We'll use ChannelReducer (a wrapper around scikit learn's factorization tools)
        # to apply Non-Negative Matrix factorization (NMF).

        nmf = ChannelReducer(n_groups, "NMF")
        spatial_factors = nmf.fit_transform(acts)[0].transpose(2, 0, 1).astype("float32")
        channel_factors = nmf._reducer.components_.astype("float32")

        # Let's organize the channels based on their horizontal position in the image

        x_peak = np.argmax(spatial_factors.max(1), 1)
        ns_sorted = np.argsort(x_peak)
        spatial_factors = spatial_factors[ns_sorted]
        channel_factors = channel_factors[ns_sorted]

        # And create a feature visualziation of each group

        param_f = lambda: param.image(80, batch=n_groups)
        obj = sum(objectives.direction(layer, channel_factors[i], batch=i)
                  for i in range(n_groups))
        group_icons = render.render_vis(model, obj, param_f, verbose=False)[-1]

        # We'd also like to know about attribution

        # First, let's turn each group into a vector over activations
        group_vecs = [spatial_factors[i, ..., None] * channel_factors[i]
                      for i in range(n_groups)]

        attrs = np.asarray([raw_class_group_attr(img, layer, attr_class, group_vecs)
                            for attr_class in attr_classes])

        print(
            attrs
        )

        # Let's render the visualization!

        with open('result/' + filename + '.html', 'a') as f:
            f.write('''  <main%s></main%s>
                          <script>
                            var app = new GroupWidget_1cb0e0d({
                              target: document.querySelector( 'main%s' ),''' % (key, key, key))
            f.write('''data: {''')
            f.write('"img":"%s",\n'%str(_image_url(img)))
            f.write('"n_groups"' + ":" + str(n_groups) + ',\n')
            f.write('"spatial_factors"' + ":" + str(
                [_image_url(factor[..., None] / np.percentile(spatial_factors, 99) * [1, 0, 0]) for factor in
                 spatial_factors]) + ',\n')
            f.write('"group_icons"' + ":" + str([_image_url(icon) for icon in group_icons]) + ',\n')
            f.write('''} });''')
            f.write('''</script>''')

    with open('result/' + filename+'.html', 'a') as f:
        f.write('''</body></html >''')
    print(filename)
imglist=[]
img = load("rar.png")
img = img[:, :, 0:3]
imglist.append(img)
img = load("adv.png")
img = img[:, :, 0:3]
imglist.append(img)
filenamelist = ['rar','adv']


neuron_groups(imglist, filenamelist, "mixed5a", 2, ["guacamole"])
# 4d
# tabby
# [[ 0.74442852  0.91507626 -1.47238791  0.45166963  0.42995349  1.96742225
#    1.36328828  2.02877903  2.45953035 -0.94934189  1.11171043  1.10638499
#    0.04262164  0.23066241  1.62526214  0.4787069   0.6795724   0.66656137]]
# adv_tabby
# [[ 0.74019086  0.80981618  0.52197969  0.79553312  1.85799694  0.53119451
#    1.37018788  0.39277077  1.71987665  2.58694148  0.25573224  0.85892642
#   -1.35404253  1.81914413  1.73091662  0.27351204  0.38520172 -1.72054458]]
# guacamole
# [[ 0.1335488   0.21030641 -0.80043489  0.06769263  0.21869409 -0.74621248
#    0.22617681 -0.12130944  0.36734137  0.73916131 -0.06092065  0.94749385
#   -0.4751839   0.0404107   0.37815315 -0.12266797 -0.61753893  0.02945981]]
