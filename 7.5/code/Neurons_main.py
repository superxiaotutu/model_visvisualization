import base64

import cv2
import numpy as np
import requests
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
from lucid.misc.io.serialize_array import serialize_array
from lucid.misc.io.showing import _image_url
from lucid.misc.gradient_override import gradient_override_map
import os

model = models.InceptionV1()
model.load_graphdef()
labels_str = read("map_clsloc.txt", encoding='utf8')
labels = [line[line.find(" "):].strip() for line in labels_str.split("\n")]
labels = [label[label.find(" "):].strip().replace("_", " ") for label in labels]
labels = ["dummy"] + labels


def raw_class_group_attr(img, layer, label, group_vecs, override=None, ):
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

def to_image_url(array, fmt='png', mode="data", quality=70, domain=None):
  supported_modes = ("data")
  if mode not in supported_modes:
    message = "Unsupported mode '%s', should be one of '%s'."
    raise ValueError(message, mode, supported_modes)
  image_data = serialize_array(array, fmt=fmt, quality=quality)
  base64_byte_string = base64.b64encode(image_data).decode('ascii')
  return base64_byte_string
def neuron_groups(img, filename, layer, n_groups=10, attr_classes=None, filenumber=0):
    # Compute activations
    dirname = '../images/' + filename+'/'
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
    try:
        os.mkdir(dirname )

    except Exception as e:
        print(e)
    # Let's render the visualization!
    finally:
        with open(dirname + '/attrs.txt', 'w') as f_w:
            f_w.write(str(attrs))
        for index, icon in enumerate(group_icons):
            imgdata=to_image_url(icon)
            print(imgdata)
            imgdata = base64.b64decode(str(imgdata))
            print(imgdata)
            with open(dirname + str(index) + '.png', 'wb') as f_jpg:
                f_jpg.write(imgdata)
                # with open(dirname+'' + '.html', 'w') as f:
                #     f.write('''<!DOCTYPE html>
                #                 <html>
                #                 <head >
                #                 <h1>%s </h1>
                #                   <title>特征可视化</title>
                #                       <script src='GroupWidget_1cb0e0d.js'></script>
                #                 </head>
                #                 <body>
                #                   <main></main>
                #                   <h1>laogewenle </h1>
                #                   <script>
                #                     var app = new GroupWidget_1cb0e0d({
                #                       target: document.querySelector( 'main' ),''' % filename)
                #     f.write('''      data: {
                #      ''')
                #     f.write('"img"' + ':"' + str(_image_url(img)) + '",\n')
                #
                #     f.write('"n_groups"' + ":" + str(n_groups) + ',\n')
                #     f.write('"spatial_factors"' + ":" + str(
                #         [_image_url(factor[..., None] / np.percentile(spatial_factors, 99) * [1, 0, 0]) for factor in
                #          spatial_factors]) + ',\n')
                #     f.write('"group_icons"' + ":" + str([_image_url(icon) for icon in group_icons]) + ',\n')
                #     f.write('''} });''')
                #     f.write('''</script>
                #             </body>
                #             </html>''')


def download():
    with open('../sources/list') as f:
        limit = 250
        for i in range(limit):
            text = f.readline()
            res = requests.get(text)
            res.raw.decode_content = True
            if res.status_code == 200:
                print(i)
                with open('../sources/' + str(i) + '.jpg', 'wb') as image_f:
                    image_f.write(res.content)


if __name__ == '__main__':
    # download()
    # for roots, dirs, files in os.walk('../sources'):
    #     for index, f in enumerate(files):
    #         print(f)
    #         img = load('../sources/' + f)
    #         img = img[:, :, 0:3]
    #         neuron_groups(img, str(index), "mixed5a", 2, ["Persian_cat"], filenumber=0)
    #         break
    for roots  in range(1,20):


    img = load("test.jpg")
    # filename='tutu'
    # img = img[:,:,0:3]
    # neuron_groups(img, filename, "mixed5a", 2, ["guacamole"],filenumber=0)
    print('x01qwe3赖美云 https://www.duweas.com/laimeiyun/')

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
