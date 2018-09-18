import json

import numpy as np
import PIL
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

import sys

def load_abc_type_img(path, box_l):
    I = PIL.Image.open(path)
    A = I.resize((299, 299)).crop((0, 0, 299, 299))
    A = (np.asarray(A) / 255.0).astype(np.float32)
    mengban = np.ones((299, 299, 3))
    width_change, length_change = 299/np.asarray(I).shape[0], 299/np.asarray(I).shape[1]

    size_of_mengban = 0
    for bx in box_l:
        mengban[int(bx[1]*width_change): int((bx[1]+bx[3])*width_change), int(bx[0]*length_change):int((bx[0]+bx[2])*length_change), :] = 0
        size_of_mengban += np.abs((int(bx[1]*width_change) - int((bx[1]+bx[3])*width_change))*(int(bx[0]*length_change) - int((bx[0]+bx[2])*length_change)))
    B = A*np.abs(mengban-1)
    C = A*mengban

    """test"""
    A = A*255
    B = B*255
    C = C*255

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
    return A.tobytes(), B.tobytes(), C.tobytes(), size_of_mengban/(299*299)

def showAnns(ann):
    ax = plt.gca()
    plt.axis('off')
    ax.set_autoscale_on(False)
    polygons = []
    for seg in ann['segmentation']:
        poly = np.array(seg).reshape((int(len(seg)/2), 2))
        polygons.append(Polygon(poly))
    p = PatchCollection(polygons, facecolor=[1,1,1], linewidths=10, alpha=1)
    ax.add_collection(p)

# def load_json(img_id):
#     cat_id_list = set()
#     img_dir = "/run/media/kirin/DOC/ImageNet2012/coco2017/train2017/"
#     return_pack = {}
#     img = coco.loadImgs(img_id)[0]
#     image_url = img_dir+img['file_name']
#
#     ann_ids=coco.getAnnIds(imgIds=img_id)
#     for ann_id in ann_ids:
#         anns = coco.loadAnns(ann_id)
#         cat_id_list.add(anns[0]['category_id'])
#     for cat_id in cat_id_list:
#         anns_list=[]
#         ann_ids = coco.getAnnIds(imgIds=img_id,catIds=cat_id)
#         for ann_id in ann_ids:
#             anns = coco.loadAnns(ann_id)
#             anns_list.append(np.round(anns[0]['bbox']))
#         img_A, img_B, img_C, size_of_m =load_abc_type_img(image_url, anns_list)
#         return_pack.update({size_of_m: [cat_id, img_A, img_B, img_C]})
#     max_label_pack = return_pack[sorted(return_pack.keys(), reverse=True)[0]]
#     return [[max_label_pack[0], max_label_pack[1]]]


annFile = 'instances_val2017.json'

img_str='{"license": 2,"file_name": "000000000139.jpg"' \
        ',"coco_url": "http://images.cocodataset.org/val2017/000000000139.jpg"' \
        ',"height": 426,"width": 640,"date_captured": "2013-11-21 01:34:01","flickr_url":' \
        ' "http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg","id": 139}'
img=plt.imread('000000000139.jpg')
plt.imshow(img)
ann_str=json.loads('{"segmentation": [[9.66,167.76,156.35,173.04,153.71,256.48,82.56,262.63,7.03,260.87]],' \
        '"area": 13244.657700000002,' \
        '"iscrowd": 0,"image_id": 139,"bbox": [7.03,167.76,149.32,94.87],"category_id": 72,"id": 34646}'
                  )
showAnns(ann_str)
fig = plt.gcf()
canvas = fig.canvas
canvas.draw()
data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
plt.close()
# print(np.where(data==[0,0,0]))
# data=np.where(data == [0, 0, 0], [1, 1, 1], [0, 0, 0])
plt.imshow(data)
print(data)

plt.show()
ann1_str=json.loads('{"segmentation": [[37.31,373.02,57.4,216.61,67.44,159.21,77.49,113.29,91.84,86.03,123.41,84.59,162.15,96.07,215.25,86.03,261.17,70.24,285.56,68.81,337.22,68.81,411.84,93.2,454.89,107.55,496.5,255.35,513.72,262.53,552.47,292.66,586.0,324.23,586.0,381.63,586.0,449.08,586.0,453.38,578.3,616.97,518.03,621.27,444.84,624.14,340.09,625.58,136.32,625.58,1.43,632.75,7.17,555.26,5.74,414.64]],"area": 275709.8110500001,"iscrowd": 0,"image_id": 285,"bbox": [1.43,68.81,584.57,563.94],"category_id": 23,"id": 587562}')
img=plt.imread('000000000285.jpg')
# plt.imshow(img)
# showAnns(ann_str)
