import random
from PIL import Image
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO


def load_abc_type_img(path, box_l):
    I = Image.open(path)
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

    return A.tobytes(), B.tobytes(), C.tobytes(), size_of_mengban / (299 * 299)


def save_my_fig(imgname, imgsize):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    ax.axis('off')
    dpi = 400
    w = imgsize[0] / dpi
    h = imgsize[1] / dpi
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.gcf().set_size_inches(w, h)
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig(imgname, format='jpg', transparent=True, dpi=dpi)
    plt.close()


def paint_ann(ann):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    plt.axis('off')
    polygons = []
    for seg in ann['segmentation']:
        poly = np.array(seg).reshape((int(len(seg) / 2), 2))
        polygons.append(Polygon(poly))
    p = PatchCollection(polygons, edgecolors=[0, 0, 0], facecolor=[0, 0, 0], linewidths=5, alpha=1)
    ax.add_collection(p)


def get_new_img(front_path, back_path, anns):
    new_img = []
    mask_front_file = "mask_front.jpg"
    mask_front_all1_file = "mask_front_all1.jpg"
    mask_back_file = "mask_back.jpg"
    new_img_file = "new_img.jpg"

    for index, ann in enumerate(anns):
        front_img = Image.open(front_path)
        back_img = Image.open(back_path)
        front_img_size = front_img.size
        back_img_size = front_img_size

        back_img = back_img.resize((front_img_size[0], front_img_size[1])).crop(
            (0, 0, front_img_size[0], front_img_size[1]))

        plt.imshow(np.asarray(front_img))
        paint_ann(ann)
        save_my_fig(mask_front_file, front_img_size)

        plt.imshow(np.asarray(back_img))

        paint_ann(ann)
        save_my_fig(mask_back_file, back_img_size)

        mask_front_all1 = np.ones((front_img_size[1], front_img_size[0], 3))
        plt.imshow(mask_front_all1)
        paint_ann(ann)
        save_my_fig(mask_front_all1_file, front_img_size)
        mask_front_all1 = (np.asarray(Image.open(mask_front_all1_file)))
        mask_front_all1 = np.where(mask_front_all1 == 255, [1, 1, 1], mask_front_all1)
        mask_front_all1 = np.abs(mask_front_all1 - 1)
        mask_front_img = (np.asarray(Image.open(front_path)))
        new_front_img = mask_front_img * mask_front_all1

        new_back_img = Image.open(mask_back_file)
        new_back_img = new_back_img.resize((front_img_size[0], front_img_size[1]))
        new_back_img = np.asarray(new_back_img)

        new_img = new_front_img + new_back_img
        plt.imshow(new_img)
        save_my_fig(new_img_file, front_img_size)
        new_img = Image.open(new_img_file)

        plt.imshow(new_front_img)
        plt.show()
        plt.imshow(new_back_img)
        plt.show()
        plt.imshow(new_img)
        plt.show()
        plt.close()
    return new_img


dataDir = "J:/9.15/annotations_trainval2017/"
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
img_path = "J:/9.15/val2017/"
coco = COCO(annFile)

# imgcatIds = coco.getCatIds()[0]
# imgids = coco.getImgIds()
# imgid = coco.getImgIds(catIds=imgcatIds)[0]
# imgannids = coco.getAnnIds(imgid, imgcatIds)
# anns=coco.loadAnns(imgannids)

# front_path = img_path + coco.loadImgs(imgid)[0]['file_name']
# back_path = img_path + coco.loadImgs(imgids[random.randint(0,2)])[0]['file_name']

imgid = "000000000724.jpg"
imgid1 = "000000001296.jpg"
front_path = img_path + imgid
back_path = img_path + imgid1
annids = coco.getAnnIds(724)
anns = coco.loadAnns(annids)
print(anns)
get_new_img(front_path, back_path, anns)




#
# dataDir = "/run/media/kirin/DOC/ImageNet2012/coco2017/annotations_trainval2017/"
# dataType = 'val2017'
# annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
# coco = COCO(annFile)
# imgcatIds = coco.getCatIds()
# imgids=coco.getImgIds()
# imganns=coco.getAnnIds()
# print(len(imganns), len(imgids), len(imgcatIds))
