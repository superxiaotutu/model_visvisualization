import numpy as np
import PIL
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from pycocotools.coco import COCO
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

def load_json(img_id):
    cat_id_list = set()
    img_dir = "/run/media/kirin/DOC/ImageNet2012/coco2017/val2017/"
    return_pack = {}
    img = coco.loadImgs(img_id)[0]
    image_url = img_dir+img['file_name']

    ann_ids=coco.getAnnIds(imgIds=img_id)
    for ann_id in ann_ids:
        anns = coco.loadAnns(ann_id)
        cat_id_list.add(anns[0]['category_id'])
    for cat_id in cat_id_list:
        anns_list=[]
        ann_ids = coco.getAnnIds(imgIds=img_id,catIds=cat_id)
        for ann_id in ann_ids:
            anns = coco.loadAnns(ann_id)
            anns_list.append(np.round(anns[0]['bbox']))
        img_A, img_B, img_C, size_of_m =load_abc_type_img(image_url, anns_list)
        return_pack.update({size_of_m: [cat_id, img_A, img_B, img_C]})
    # for i in return_pack:
    #     plt.imshow(i[2]/2+0.5)
    #     plt.show()
    max_label_pack = return_pack[sorted(return_pack.keys(), reverse=True)[0]]
    return [[max_label_pack[0], max_label_pack[1]]]

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

writer = tf.python_io.TFRecordWriter('/run/media/kirin/DOC/ImageNet2012/test.tfrecords')
dataDir = "/run/media/kirin/DOC/ImageNet2012/coco2017/annotations_trainval2017/"
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)
imgIds = coco.getImgIds()
img_ids = coco.loadImgs(imgIds)

tutu = 0
err = 0
for img in img_ids:
    try:
        img_id=img['id']
        data_pack = load_json(img_id)
        for data in data_pack:
            feature = {'label': _int64_feature(data[0]), 'test_img': _bytes_feature(data[1])}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    except:
        err += 1
        print("\t\tERRO!!!!!!!!")
    print(tutu/len(img_ids))
    tutu += 1

print("ERRO:"+str(err))
