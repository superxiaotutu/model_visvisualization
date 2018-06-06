'''
!pip install --quiet lucid==0.0.5
!npm install -g svelte-cli@2.2.0
svelte compile --format iife SpatialWidget_3725625.html > SpatialWidget_3725625.js
'''
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
labels_str = read("map_clsloc.txt",encoding='utf8')
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


def neuron_groups(img, filename,layer, n_groups=6, attr_classes=None):
    # Compute activations

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
    with open('results_'+filename+'.html', 'w') as f:
        f.write('''<!DOCTYPE html>
                    <html>
                    <head >
                      <title>My first Svelte app</title>
                          <script src='GroupWidget_1cb0e0d.js'></script>
                    </head>
                    <body>
                      <main></main>
                      <h1>laogewenle </h1>
                      <script>
                        var app = new GroupWidget_1cb0e0d({
                          target: document.querySelector( 'main' ),''')
        f.write('''      data: {
         ''')
        f.write('"img"' + ':"' + str(_image_url(img)) + '",\n')

        f.write('"n_groups"' + ":" + str(n_groups) + ',\n')
        f.write('"spatial_factors"' + ":" + str( [_image_url(factor[..., None] / np.percentile(spatial_factors, 99) * [1, 0, 0]) for factor in
                            spatial_factors]) + ',\n')
        f.write('"group_icons"' + ":" + str([_image_url(icon) for icon in group_icons]) + ',\n')
        f.write('''} });''')
        f.write('''</script>
                </body>
                </html>''')


# img = cv2.imread('new_adv.png')
# img = cv2.resize(img, (224, 224))
# cv2.imwrite('new_adv.png',img)
img = load("new_rar.png")
filename='rar'
neuron_groups(img, filename, "mixed4d", 6, ["tabby"])
# neuron_groups(img, filename,"mixed4d", 6, ["Labrador retriever", "tiger cat"])


# hint_label_1="Labrador retriever", hint_label_2="tiger cat"
