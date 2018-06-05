import numpy as np
import tensorflow as tf
from IPython import get_ipython
import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.render as render
from lucid.misc.io import show, load
from lucid.misc.io.reading import read
from lucid.misc.io.showing import _image_url
from lucid.misc.gradient_override import gradient_override_map

# import lucid.scratch.web.svelte as lucid_svelte

model = models.InceptionV1()
model.load_graphdef()
labels_str = read("https://gist.githubusercontent.com/aaronpolhamus/964a4411c0906315deb9f4a3723"
                  "aac57/raw/aa66dd9dbf6b56649fa3fab83659b2acbf3cbfd1/map_clsloc.txt", encoding='utf8')

labels = [line[line.find(" "):].strip() for line in labels_str.split("\n")]
labels = [label[label.find(" "):].strip().replace("_", " ") for label in labels]
labels = ["dummy"] + labels


def raw_class_spatial_attr(img, layer, label, override=None):
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
        return np.sum(acts * grad, -1)[0]


def raw_spatial_spatial_attr(img, layer1, layer2, override=None):
    """Attribution between spatial positions in two different layers."""

    # Set up a graph for doing attribution...
    with tf.Graph().as_default(), tf.Session(), gradient_override_map(override or {}):
        t_input = tf.placeholder_with_default(img, [None, None, 3])
        T = render.import_model(model, t_input, t_input)

        # Compute activations
        acts1 = T(layer1).eval()
        acts2 = T(layer2).eval({T(layer1): acts1})

        # Construct gradient tensor
        # Backprop from spatial position (n_x, n_y) in layer2 to layer1.
        n_x, n_y = tf.placeholder("int32", []), tf.placeholder("int32", [])
        layer2_mags = tf.sqrt(tf.reduce_sum(T(layer2) ** 2, -1))[0]
        score = layer2_mags[n_x, n_y]
        t_grad = tf.gradients([score], [T(layer1)])[0]

        # Compute attribution backwards from each positin in layer2
        attrs = []
        for i in range(acts2.shape[1]):
            attrs_ = []
            for j in range(acts2.shape[2]):
                grad = t_grad.eval({n_x: i, n_y: j, T(layer1): acts1})
                # linear approximation of imapct
                attr = np.sum(acts1 * grad, -1)[0]
                attrs_.append(attr)
            attrs.append(attrs_)
    return np.asarray(attrs)


def orange_blue(a, b, clip=False):
    if clip:
        a, b = np.maximum(a, 0), np.maximum(b, 0)
    arr = np.stack([a, (a + b) / 2., b], -1)
    arr /= 1e-2 + np.abs(arr).max() / 1.5
    arr += 0.3
    return arr


def image_url_grid(grid):
    return [[_image_url(img) for img in line] for line in grid]


def spatial_spatial_attr(img, layer1, layer2, hint_label_1=None, hint_label_2=None, override=None):
    hint1 = orange_blue(
        raw_class_spatial_attr(img, layer1, hint_label_1, override=override),
        raw_class_spatial_attr(img, layer1, hint_label_2, override=override),
        clip=True
    )
    hint2 = orange_blue(
        raw_class_spatial_attr(img, layer2, hint_label_1, override=override),
        raw_class_spatial_attr(img, layer2, hint_label_2, override=override),
        clip=True
    )

    attrs = raw_spatial_spatial_attr(img, layer1, layer2, override=override)
    attrs = attrs / attrs.max()

    with open('results.html', 'w') as f:
        f.write('''<!DOCTYPE html>
                    <html>
                    <head >
                      <title>My first Svelte app</title>
                          <script src='SpatialWidget_141d66.js'></script>
                    </head>
                    <body>
                      <main></main>
                      <h1>laogewenle </h1>
                      <script>
                        var app = new SpatialWidget_141d66({
                          target: document.querySelector( 'main' ),''')
        f.write('''      data: {
          "layer2": "''' + layer2 + '''",
          "layer1": "''' + layer1 + '''",''')
        f.write('"spritemap1"' + ":" + str(image_url_grid(attrs)) + ',\n')
        f.write('"spritemap2"' + ":" + str(image_url_grid(attrs.transpose(2, 3, 0, 1))) + ',\n')
        f.write('"size1"' + ":" + str(attrs.shape[3]) + ',\n')
        f.write('"size2"' + ":" + str(attrs.shape[0]) + ',\n')
        f.write('"img"' + ":" + '"' + str(_image_url(img)) + '"' + ',\n')
        f.write('"hint1"' + ":" + '"' + str(_image_url(hint1)) + '"' + ',\n')
        f.write('"hint2"' + ":" + '"' + str(_image_url(hint2)) + '"' + '\n')
        f.write('''} });''')
        f.write('''</script>
                </body>
                </html>''')

        # lucid_svelte.SpatialWidget({
        #     "spritemap1": image_url_grid(attrs),
        #     "spritemap2": image_url_grid(attrs.transpose(2, 3, 0, 1)),
        #     "size1": attrs.shape[3],
        #     "layer1": layer1,
        #     "size2": attrs.shape[0],
        #     "layer2": layer2,
        #     "img": _image_url(img),
        #     "hint1": _image_url(hint1),
        #     "hint2": _image_url(hint2)
        # })


import cv2

img = cv2.imread('new_dog.jpg')
# img = cv2.resize(img, (224, 224))
# cv2.imwrite('new_dog',img)
img = load('new_dog.jpg')

spatial_spatial_attr(img, "mixed3a", "mixed5b", hint_label_1="Siberian husky")

print("\nHover on images to interact! :D\n")
