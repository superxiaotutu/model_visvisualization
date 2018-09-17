import PIL
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from mxnet import image
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO


def load_abc_type_img(path, box_l):
    I = PIL.Image.open(path)
    A = I.resize((299, 299)).crop((0, 0, 299, 299))
    A = (np.asarray(A) / 255.0).astype(np.float32)
    mengban = np.ones((299, 299, 3))
    width_change, length_change = 299 / np.asarray(I).shape[0], 299 / np.asarray(I).shape[1]

    size_of_mengban = 0
    for bx in box_l:
        mengban[int(bx[1] * width_change): int((bx[1] + bx[3]) * width_change),
        int(bx[0] * length_change):int((bx[0] + bx[2]) * length_change), :] = 0
        size_of_mengban += np.abs((int(bx[1] * width_change) - int((bx[1] + bx[3]) * width_change)) * (
        int(bx[0] * length_change) - int((bx[0] + bx[2]) * length_change)))
    B = A * np.abs(mengban - 1)
    C = A * mengban

    """test"""
    A = A * 255
    B = B * 255
    C = C * 255

    A = A.astype(np.uint8)
    B = B.astype(np.uint8)
    C = C.astype(np.uint8)

    # plt.imshow(A)
    # plt.show()
    # plt.imshow(mengban)
    # plt.show()
    # plt.imshow(B)
    # plt.show()
    # plt.imshow(C)
    # plt.show()
    return A.tobytes(), B.tobytes(), C.tobytes(), size_of_mengban / (299 * 299)


image_file = "J:/9.15/val2017"
ana_dir = "J:/9.15/annotations_trainval2017/annotations/"
coco = COCO(ana_dir + "instances_val2017.json")
# print(coco.getAnnIds(285))
# ann=coco.loadAnns(587562)
with open("J:/9.15/val2017/000000000285.jpg", 'rb') as f:
    I = image.imdecode(f.read()).asnumpy()

plt.imshow(I)
# plt.show()

# annIds=coco.getAnnIds(285)
# anns = coco.loadAnns(587562)
anns = [{'segmentation': [
    [37.31, 373.02, 57.4, 216.61, 67.44, 159.21, 77.49, 113.29, 91.84, 86.03, 123.41, 84.59, 162.15, 96.07, 215.25,
     86.03, 261.17, 70.24, 285.56, 68.81, 337.22, 68.81, 411.84, 93.2, 454.89, 107.55, 496.5, 255.35, 513.72, 262.53,
     552.47, 292.66, 586.0, 324.23, 586.0, 381.63, 586.0, 449.08, 586.0, 453.38, 578.3, 616.97, 518.03, 621.27, 444.84,
     624.14, 340.09, 625.58, 136.32, 625.58, 1.43, 632.75, 7.17, 555.26, 5.74, 414.64]], 'area': 275709.8110500001,
         'iscrowd': 0, 'image_id': 285, 'bbox': [1.43, 68.81, 584.57, 563.94], 'category_id': 23, 'id': 587562}]
def paint_ann(ann):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for seg in ann['segmentation']:
        poly = np.array(seg).reshape((int(len(seg) / 2), 2))
        polygons.append(Polygon(poly))
    p = PatchCollection(polygons, facecolor=[0, 0, 0], linewidths=0.5, alpha=1)
    ax.add_collection(p)
    file_name="a.png"
    plt.savefig(file_name)
    return file_name
def get_front_mask(path,box_list):
    I = PIL.Image.open(path)
    A = I.resize((299, 299)).crop((0, 0, 299, 299))
    A = (np.asarray(A) / 255.0).astype(np.float32)
    mengban = np.ones((299, 299, 3))
    width_change, length_change = 299 / np.asarray(I).shape[0], 299 / np.asarray(I).shape[1]
    size_of_mengban = 0
    for bx in box_list:
        mengban[int(bx[1] * width_change): int((bx[1] + bx[3]) * width_change),
        int(bx[0] * length_change):int((bx[0] + bx[2]) * length_change), :] = 0

        size_of_mengban += np.abs((int(bx[1] * width_change) - int((bx[1] + bx[3]) * width_change)) * (
            int(bx[0] * length_change) - int((bx[0] + bx[2]) * length_change)))
        # plt.imshow(A)
        # plt.show()
    # B = A * np.abs(mengban - 1)
    # C = A * mengban
load_abc_type_img(paint_ann(anns[0]),anns[0])




#
# dataDir = "/run/media/kirin/DOC/ImageNet2012/coco2017/annotations_trainval2017/"
# dataType = 'val2017'
# annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
# coco = COCO(annFile)
# imgcatIds = coco.getCatIds()
# imgids=coco.getImgIds()
# imganns=coco.getAnnIds()
# print(len(imganns), len(imgids), len(imgcatIds))
