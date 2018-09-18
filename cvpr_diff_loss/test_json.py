import random
from PIL import Image
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np


# from pycocotools.coco import COCO


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
    fig = plt.gcf()
    canvas = fig.canvas
    canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data

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
        #对img_front和img_back分割。
        plt.imshow(np.asarray(front_img))
        paint_ann(ann)
        save_my_fig(mask_front_file, front_img_size)
        plt.imshow(np.asarray(back_img))
        paint_ann(ann)
        save_my_fig(mask_back_file, back_img_size)

        # 新建全1图，并且为他画上分割。
        mask_front_all1 = np.ones((front_img_size[1], front_img_size[0], 3))
        plt.imshow(mask_front_all1)
        paint_ann(ann)
        save_my_fig(mask_front_all1_file, front_img_size)

        # 把全1图化成标记为1其他部分为0
        mask_front_all1 = np.asarray(Image.open(mask_front_all1_file))
        mask_front_all1 = np.clip(mask_front_all1, 0, 1)
        mask_front_all1 = np.abs(mask_front_all1 - 1)
        mask_front_all1 = np.clip(mask_front_all1, 0, 1)

        # 把前景的原图片图片乘以标记部分为1的矩阵，得到只有标记部分的图片
        front_img = np.asarray(Image.open(front_path))
        new_front_img = front_img * mask_front_all1

        # 把扣掉标签的背景图片加上只有标记部分的图片
        new_back_img = Image.open(mask_back_file)
        new_back_img = new_back_img.resize((front_img_size[0], front_img_size[1]))
        new_back_img = np.asarray(new_back_img)
        new_img = new_front_img + new_back_img

        plt.imshow(new_front_img)
        plt.show()
        plt.imshow(new_back_img)
        plt.show()
        plt.imshow(new_img)
        plt.show()
        plt.close()
    return new_img


anns_str = [{'segmentation': [
    [426.91, 58.24, 434.49, 77.74, 467.0, 80.99, 485.42, 86.41, 493.0, 129.75, 521.17, 128.67, 532.01, 144.92, 545.01,
     164.42, 552.6, 170.93, 588.35, 178.51, 629.53, 165.51, 629.53, 177.43, 578.6, 214.27, 558.01, 241.35, 526.59,
     329.12, 512.51, 370.29, 502.75, 415.8, 418.24, 409.3, 399.82, 414.72, 388.98, 420.14, 382.48, 424.47, 391.15,
     430.97, 414.99, 425.55, 447.49, 427.72, 449.66, 435.3, 431.24, 438.56, 421.49, 452.64, 422.57, 456.98, 432.33,
     464.56, 439.91, 458.06, 481.08, 465.64, 502.75, 464.56, 507.09, 473.23, 639.28, 474.31, 639.28, 1.9, 431.24, 0.0]],
             'area': 63325.421899999994, 'iscrowd': 0, 'image_id': 522418, 'bbox': [382.48, 0.0, 256.8, 474.31],
             'category_id': 1, 'id': 455475}, {'segmentation': [
    [416.41, 449.28, 253.36, 422.87, 234.06, 412.2, 277.23, 406.61, 343.77, 411.69, 379.84, 414.23, 384.41, 424.9,
     397.11, 427.95, 410.31, 427.95, 445.36, 429.98, 454.0, 438.61, 431.65, 438.61, 423.01, 449.28]],
                                               'area': 4200.516899999997, 'iscrowd': 0, 'image_id': 522418,
                                               'bbox': [234.06, 406.61, 219.94, 42.67], 'category_id': 49,
                                               'id': 692513}, {'segmentation': [
    [71.19, 327.91, 5.39, 371.06, 0.0, 371.06, 0.0, 473.53, 365.66, 473.53, 379.69, 442.25, 354.88, 431.46, 247.01,
     417.44, 232.99, 410.97, 277.21, 406.65, 326.83, 408.81, 379.69, 416.36, 386.16, 418.52, 393.71, 413.12, 406.65,
     379.69, 406.65, 366.74, 399.1, 339.78, 286.92, 323.6, 179.06, 318.2, 98.16, 316.04]], 'area': 54409.19939999999,
                                                               'iscrowd': 0, 'image_id': 522418,
                                                               'bbox': [0.0, 316.04, 406.65, 157.49], 'category_id': 61,
                                                               'id': 1085508},
            {'segmentation': [[347.84, 225.66, 311.69, 249.35, 305.45, 205.71, 361.56, 172.05, 362.81, 179.53]],
             'area': 2220.645799999999, 'iscrowd': 0, 'image_id': 522418, 'bbox': [305.45, 172.05, 57.36, 77.3],
             'category_id': 81, 'id': 1982455}]
anns1_str = [{'segmentation': [
    [255.07, 116.22, 265.91, 111.16, 283.97, 113.32, 338.89, 114.77, 347.56, 115.49, 363.46, 128.5, 368.52, 143.67,
     366.35, 159.57, 364.18, 166.8, 362.74, 178.36, 337.45, 194.25, 334.55, 194.98, 331.66, 184.86, 329.5, 179.8,
     324.44, 174.02, 276.75, 170.41, 265.91, 163.18, 265.19, 158.85, 251.46, 144.4, 244.95, 137.89, 239.9, 122.72,
     243.51, 119.11],
    [276.75, 182.69, 277.47, 196.42, 280.36, 205.09, 270.25, 210.15, 268.8, 210.87, 264.46, 210.87, 268.08, 202.2,
     271.69, 193.53, 267.35, 179.8]], 'area': 7142.43465, 'iscrowd': 0, 'image_id': 184613,
              'bbox': [239.9, 111.16, 128.62, 99.71], 'category_id': 21, 'id': 72124}, {'segmentation': [
    [285.08, 109.72, 310.36, 94.24, 336.15, 91.15, 341.31, 93.21, 354.72, 89.09, 386.19, 92.18, 417.14, 87.54, 437.26,
     85.99, 455.31, 109.2, 448.09, 127.26, 444.99, 136.03, 418.69, 143.25, 416.11, 137.06, 402.18, 148.92, 376.39,
     150.47, 368.65, 146.34, 364.52, 131.9, 348.01, 118.49, 333.57, 112.81, 292.3, 112.81]], 'area': 6260.133750000001,
                                                                                        'iscrowd': 0,
                                                                                        'image_id': 184613,
                                                                                        'bbox': [285.08, 85.99, 170.23,
                                                                                                 64.48],
                                                                                        'category_id': 21, 'id': 75495},
             {'segmentation': [
                 [473.23, 91.56, 481.38, 89.49, 495.01, 86.67, 500.0, 86.82, 500.0, 95.12, 493.08, 108.75, 490.56,
                  103.27, 494.86, 96.45, 494.42, 93.19, 490.86, 91.56, 488.05, 92.3, 484.49, 94.38, 482.12, 99.12,
                  472.19, 100.6, 469.82, 101.34, 469.53, 99.86, 460.19, 103.56, 457.08, 104.01, 452.49, 101.78, 464.05,
                  90.67, 466.56, 88.9, 467.45, 85.93, 470.86, 88.01, 476.19, 88.9]], 'area': 451.0984499999998,
              'iscrowd': 0, 'image_id': 184613, 'bbox': [452.49, 85.93, 47.51, 22.82], 'category_id': 21, 'id': 75654},
             {'segmentation': [
                 [299.26, 71.84, 302.51, 69.16, 306.34, 68.01, 308.83, 70.69, 313.04, 72.61, 313.23, 76.05, 312.65,
                  84.09, 316.48, 86.96, 317.63, 80.07, 319.35, 81.03, 318.59, 88.49, 315.52, 89.07, 308.44, 88.49,
                  306.53, 87.73, 303.28, 87.92, 301.36, 88.11, 300.02, 85.24, 298.11, 81.8, 296.96, 78.54, 297.72,
                  76.05, 299.45, 72.23]], 'area': 286.82354999999995, 'iscrowd': 0, 'image_id': 184613,
              'bbox': [296.96, 68.01, 22.39, 21.06], 'category_id': 21, 'id': 278765}, {'segmentation': [
        [490.06, 84.64, 483.77, 87.47, 482.35, 87.47, 479.92, 88.49, 477.29, 90.11, 476.48, 90.11, 475.46, 88.89,
         474.24, 87.07, 473.43, 87.07, 472.01, 86.66, 468.97, 85.04, 466.74, 83.62, 464.92, 84.64, 462.49, 84.03,
         461.07, 83.22, 461.68, 80.38, 462.28, 78.35, 464.31, 78.96, 466.54, 77.74, 468.37, 76.53, 469.58, 76.93,
         471.61, 77.95, 472.83, 78.35, 473.84, 78.35, 476.88, 76.73, 481.14, 76.12, 483.57, 76.12, 489.45, 75.92,
         491.48, 76.32, 495.33, 82.0, 495.53, 84.43, 493.71, 86.05]], 'area': 305.33645, 'iscrowd': 0,
                                                                                        'image_id': 184613,
                                                                                        'bbox': [461.07, 75.92, 34.46,
                                                                                                 14.19],
                                                                                        'category_id': 21,
                                                                                        'id': 278913}, {
                 'segmentation': [
                     [138.18, 31.01, 103.44, 78.58, 114.77, 94.43, 122.32, 111.8, 123.07, 122.37, 137.42, 123.12,
                      172.15, 140.49, 180.46, 142.76, 186.5, 145.78, 187.25, 153.33, 188.01, 160.12, 188.01, 162.39,
                      192.54, 166.16, 200.09, 163.9, 200.84, 160.12, 193.29, 159.37, 190.27, 141.25, 188.76, 126.9,
                      188.76, 118.59, 190.27, 105.0, 190.27, 97.45, 192.54, 80.09, 192.54, 77.07, 197.07, 69.52, 203.11,
                      65.74, 215.19, 65.74, 226.52, 69.52, 221.23, 109.53, 233.31, 118.59, 249.92, 115.57, 257.47,
                      92.17, 258.23, 77.82, 249.92, 61.21, 222.74, 43.09, 216.7, 35.54, 138.93, 33.27]],
                 'area': 10247.228050000002, 'iscrowd': 0, 'image_id': 184613, 'bbox': [103.44, 31.01, 154.79, 135.15],
                 'category_id': 28, 'id': 286024}, {'segmentation': [
        [72.52, 70.94, 72.52, 69.18, 73.08, 68.06, 73.96, 67.9, 74.04, 67.02, 75.8, 67.02, 76.12, 65.9, 77.24, 66.06,
         78.44, 65.18, 79.07, 66.38, 80.11, 66.7, 81.87, 67.18, 82.35, 68.14, 82.99, 67.02, 84.35, 67.1, 85.39, 65.74,
         85.97, 64.73, 83.35, 64.09, 81.9, 64.73, 79.64, 63.82, 78.28, 62.73, 76.56, 62.37, 73.3, 60.92, 72.48, 61.28,
         67.96, 59.84, 65.87, 60.56, 67.41, 62.73, 67.14, 64.91, 65.51, 67.71, 65.33, 69.16, 65.42, 70.43, 66.33, 70.25,
         66.24, 68.71, 66.51, 67.71, 67.5, 67.71, 68.05, 69.25, 68.32, 70.7, 69.41, 70.88, 69.22, 68.98, 69.5, 68.08,
         70.4, 68.26, 71.13, 69.7, 71.58, 70.52, 71.76, 71.97, 72.85, 72.15]], 'area': 103.56964999999995, 'iscrowd': 0,
                                                    'image_id': 184613, 'bbox': [65.33, 59.84, 20.64, 12.31],
                                                    'category_id': 21, 'id': 369823}, {'segmentation': [
        [218.21, 280.88, 215.95, 293.72, 203.87, 290.7, 202.36, 279.37, 202.36, 271.82, 206.13, 258.98, 207.64, 247.66,
         207.64, 237.09, 206.89, 232.56, 203.11, 225.01, 202.36, 217.46, 199.33, 216.7, 190.27, 221.23, 187.25, 233.31,
         184.99, 242.37, 181.21, 252.94, 175.17, 264.27, 169.13, 274.84, 166.11, 283.15, 165.36, 295.23, 162.34, 306.55,
         154.79, 312.59, 147.24, 303.53, 146.48, 300.51, 149.5, 293.72, 157.05, 284.66, 159.32, 277.11, 160.07, 265.02,
         160.07, 258.23, 163.85, 248.41, 167.62, 239.35, 172.15, 235.58, 174.42, 230.29, 173.66, 221.23, 172.15, 216.7,
         174.42, 206.89, 175.17, 202.36, 184.23, 184.99, 178.19, 169.13, 179.7, 161.58, 178.95, 148.75, 178.95, 145.73,
         180.46, 143.46, 176.68, 129.87, 174.42, 126.09, 174.42, 123.83, 175.17, 119.3, 178.95, 113.26, 184.99, 108.73,
         192.54, 108.73, 197.07, 107.97, 197.82, 103.44, 194.05, 97.4, 192.54, 89.85, 192.54, 80.04, 194.8, 68.71,
         203.87, 66.44, 206.13, 66.44, 210.66, 65.69, 215.19, 67.96, 221.23, 73.24, 225.01, 76.26, 225.76, 81.55,
         225.01, 84.57, 225.01, 89.1, 223.5, 92.87, 223.5, 95.14, 224.25, 104.2, 222.74, 109.48, 218.21, 109.48, 237.84,
         119.3, 245.39, 135.16, 253.7, 143.46, 256.72, 146.48, 262.0, 151.77, 268.04, 159.32, 269.56, 161.58, 274.84,
         166.11, 279.37, 171.4, 283.15, 176.68, 286.17, 178.95, 290.7, 180.46, 295.98, 181.97, 301.27, 184.23, 307.31,
         188.76, 305.04, 193.29, 296.74, 194.05, 284.66, 191.03, 280.13, 187.25, 272.58, 181.97, 264.27, 178.95, 257.47,
         174.42, 257.47, 174.42, 246.9, 168.38, 235.58, 167.62, 233.31, 181.97, 229.54, 193.29, 229.54, 204.62, 231.8,
         215.19, 228.78, 225.01, 230.29, 231.8, 225.01, 235.58, 218.21, 234.82, 216.7, 257.47, 215.95, 265.78, 215.19,
         271.07, 215.95, 279.37]], 'area': 11210.3759, 'iscrowd': 0, 'image_id': 184613, 'bbox': [146.48, 65.69, 160.83,
                                                                                                  246.9],
                                                                                       'category_id': 1, 'id': 1202318},
             {'segmentation': [
                 [57.26, 164.23, 59.52, 179.3, 63.28, 187.59, 64.79, 192.86, 69.31, 201.9, 66.3, 204.16, 53.49, 204.16,
                  46.71, 202.65, 56.5, 198.13, 56.5, 190.6, 50.48, 184.57, 48.22, 174.78, 42.94, 164.23, 39.93, 158.21,
                  33.15, 155.19, 30.13, 149.92, 30.89, 140.13, 30.13, 129.58, 24.86, 113.0, 18.83, 97.94, 21.85, 91.91,
                  23.35, 85.88, 15.07, 76.84, 8.29, 67.05, 10.55, 59.52, 23.35, 57.26, 33.15, 61.78, 39.17, 72.32, 43.7,
                  81.36, 64.79, 97.94, 69.31, 117.52, 66.3, 128.07, 66.3, 134.1, 69.31, 143.14, 70.06, 143.14, 67.05,
                  158.21, 73.08, 169.51, 76.84, 175.53, 79.86, 182.31, 82.12, 189.85, 83.62, 193.61, 74.58, 193.61,
                  71.57, 188.34, 69.31, 179.3, 64.04, 173.27, 58.01, 164.99]], 'area': 4457.008849999999, 'iscrowd': 0,
              'image_id': 184613, 'bbox': [8.29, 57.26, 75.33, 146.9], 'category_id': 1, 'id': 1236161}, {
                 'segmentation': [
                     [45.24, 48.41, 52.61, 48.41, 56.95, 48.63, 60.85, 48.85, 63.45, 50.8, 65.84, 57.3, 67.36, 63.16,
                      64.97, 68.58, 66.49, 72.7, 67.57, 73.78, 76.03, 79.42, 77.55, 81.8, 81.23, 90.26, 81.45, 94.81,
                      81.02, 104.14, 82.75, 110.64, 80.37, 117.37, 81.67, 122.14, 81.88, 126.04, 82.75, 129.94, 82.97,
                      135.8, 84.7, 140.13, 87.52, 146.85, 91.21, 151.84, 88.39, 155.53, 84.49, 157.48, 79.06, 158.13,
                      77.76, 159.21, 77.33, 160.95, 76.25, 164.85, 76.25, 167.89, 77.98, 171.36, 78.85, 173.74, 75.16,
                      173.96, 73.43, 169.19, 69.52, 163.55, 66.27, 161.82, 66.05, 157.05, 72.34, 154.23, 74.08, 152.93,
                      71.69, 147.29, 70.39, 144.04, 70.17, 137.31, 65.19, 132.54, 63.89, 130.16, 63.24, 126.26, 66.71,
                      124.74, 68.66, 120.62, 67.79, 113.25, 66.05, 105.22, 66.05, 101.54, 64.75, 98.07, 64.32, 95.68,
                      64.54, 90.26, 65.19, 85.27, 64.75, 83.54, 64.32, 80.94, 58.47, 72.91, 53.04, 69.01, 53.26, 66.41,
                      55.43, 65.33, 55.43, 60.99, 53.48, 57.52, 50.01, 53.4, 45.89, 49.93]], 'area': 1700.759,
                 'iscrowd': 0, 'image_id': 184613, 'bbox': [45.24, 48.41, 45.97, 125.55], 'category_id': 1,
                 'id': 1250302}, {'segmentation': [
        [45.71, 83.8, 39.63, 72.37, 34.99, 63.44, 20.71, 55.59, 24.99, 44.87, 38.21, 44.87, 51.42, 55.59, 54.64, 62.37,
         55.71, 70.23, 63.56, 80.94, 64.99, 95.23, 59.99, 94.52, 47.14, 83.8]], 'area': 927.7607, 'iscrowd': 0,
                                  'image_id': 184613, 'bbox': [20.71, 44.87, 44.28, 50.36], 'category_id': 1,
                                  'id': 1275450}, {'segmentation': [
        [0.75, 76.59, 1.51, 83.37, 9.04, 90.15, 12.81, 108.23, 7.53, 118.78, 20.34, 140.63, 36.91, 167.75, 36.16,
         176.04, 11.3, 179.8, 6.78, 180.56, 2.26, 182.06, 13.56, 161.72, 2.26, 136.86]], 'area': 1386.54515,
                                                   'iscrowd': 0, 'image_id': 184613,
                                                   'bbox': [0.75, 76.59, 36.16, 105.47], 'category_id': 1,
                                                   'id': 1280839}, {'segmentation': [
        [343.56, 82.1, 349.97, 76.8, 352.76, 74.01, 353.6, 70.1, 353.6, 64.8, 357.5, 63.97, 360.85, 65.64, 361.13,
         70.38, 360.85, 72.06, 362.24, 79.03, 361.41, 85.44, 360.57, 90.47, 351.09, 92.7, 349.69, 89.35, 349.69, 87.12,
         350.81, 84.61, 351.65, 82.1, 343.28, 85.72]], 'area': 297.9553000000001, 'iscrowd': 0, 'image_id': 184613,
                                                                    'bbox': [343.28, 63.97, 18.96, 28.73],
                                                                    'category_id': 1, 'id': 1281132}, {'segmentation': [
        [371.04, 73.3, 371.41, 68.9, 373.25, 66.51, 377.1, 68.34, 379.49, 73.49, 378.75, 75.14, 378.57, 79.73, 379.12,
         91.29, 370.49, 90.19, 370.68, 86.34, 365.9, 88.54, 364.8, 87.07, 367.74, 85.05, 370.86, 84.5, 371.41, 82.85,
         368.47, 81.2, 363.33, 77.89, 362.05, 76.24, 363.15, 76.24, 364.07, 74.22, 366.27, 77.34, 369.39, 80.09, 370.86,
         80.09, 373.1, 78.46, 374.29, 76.36, 375.34, 75.96, 375.21, 75.04, 373.63, 74.51]], 'area': 194.52734999999973,
                                                                                                       'iscrowd': 0,
                                                                                                       'image_id': 184613,
                                                                                                       'bbox': [362.05,
                                                                                                                66.51,
                                                                                                                17.44,
                                                                                                                24.78],
                                                                                                       'category_id': 1,
                                                                                                       'id': 1288242}, {
                 'segmentation': [
                     [383.09, 91.03, 385.06, 89.71, 385.39, 87.08, 386.05, 84.13, 386.05, 80.84, 386.05, 78.54, 385.39,
                      75.91, 385.06, 73.94, 384.08, 69.66, 383.75, 68.68, 386.05, 66.05, 388.02, 65.39, 390.32, 62.76,
                      390.98, 61.12, 394.92, 60.79, 395.58, 61.12, 393.61, 64.73, 393.61, 66.38, 394.92, 68.68, 394.6,
                      71.31, 393.61, 75.91, 392.95, 78.54, 390.98, 83.47, 392.3, 91.36, 392.62, 92.34, 388.35, 93.33,
                      387.69, 92.34, 389.01, 88.73, 389.34, 86.75, 387.37, 84.45, 386.71, 87.74, 385.39, 91.36, 382.76,
                      92.34]], 'area': 212.8566000000004, 'iscrowd': 0, 'image_id': 184613,
                 'bbox': [382.76, 60.79, 12.82, 32.54], 'category_id': 1, 'id': 1289336}, {'segmentation': [
        [420.74, 87.09, 419.18, 87.32, 418.96, 82.41, 417.62, 81.97, 416.28, 87.87, 413.61, 88.21, 414.5, 82.41, 413.61,
         81.63, 414.61, 77.17, 415.61, 72.38, 414.72, 71.38, 416.28, 67.81, 416.73, 66.47, 418.96, 65.58, 419.51, 65.69,
         418.73, 64.24, 418.84, 62.13, 421.07, 61.01, 422.3, 61.01, 423.75, 62.35, 423.64, 64.8, 423.08, 66.58, 425.2,
         67.48, 425.53, 70.6, 424.98, 74.83, 424.08, 76.84, 422.63, 82.3, 421.63, 82.64]], 'area': 200.91509999999963,
                                                                                           'iscrowd': 0,
                                                                                           'image_id': 184613,
                                                                                           'bbox': [413.61, 61.01,
                                                                                                    11.92, 27.2],
                                                                                           'category_id': 1,
                                                                                           'id': 1304428}, {
                 'segmentation': [
                     [316.07, 86.43, 321.29, 79.04, 321.94, 78.39, 321.29, 88.82, 325.64, 87.52, 326.94, 77.95, 326.51,
                      70.78, 328.03, 67.52, 326.07, 64.7, 327.16, 59.7, 325.2, 57.09, 321.29, 57.74, 321.73, 61.22,
                      314.34, 65.35, 318.25, 72.3, 316.51, 78.39]], 'area': 252.4399, 'iscrowd': 0, 'image_id': 184613,
                 'bbox': [314.34, 57.09, 13.69, 31.73], 'category_id': 1, 'id': 1309365}, {'segmentation': [
        [291.95, 86.81, 290.86, 81.39, 288.04, 77.92, 289.13, 74.02, 289.13, 68.81, 291.29, 62.96, 291.95, 60.57,
         291.73, 57.1, 296.28, 57.1, 297.37, 61.44, 296.07, 64.26, 296.5, 67.73, 297.8, 71.42, 297.15, 75.75, 295.85,
         78.35, 294.98, 79.87, 295.2, 84.64, 293.46, 86.16]], 'area': 179.7395499999998, 'iscrowd': 0,
                                                                                           'image_id': 184613,
                                                                                           'bbox': [288.04, 57.1, 9.76,
                                                                                                    29.71],
                                                                                           'category_id': 1,
                                                                                           'id': 1676698}, {
                 'segmentation': [
                     [272.72, 74.9, 273.51, 64.75, 274.31, 62.16, 273.12, 59.77, 274.31, 56.58, 276.7, 56.98, 278.29,
                      60.57, 279.09, 64.75, 279.09, 71.52, 277.69, 76.09, 279.09, 87.12, 274.11, 86.52, 274.31, 83.34,
                      274.31, 80.75, 274.31, 77.16]], 'area': 143.45519999999945, 'iscrowd': 0, 'image_id': 184613,
                 'bbox': [272.72, 56.58, 6.37, 30.54], 'category_id': 1, 'id': 1686503}, {'segmentation': [
        [15.37, 48.6, 13.3, 48.93, 12.2, 50.23, 12.2, 50.67, 12.31, 51.65, 13.62, 51.87, 12.31, 52.53, 10.13, 54.38,
         8.93, 56.23, 10.46, 57.0, 11.11, 58.42, 11.88, 59.29, 13.62, 58.96, 17.99, 58.09, 18.1, 55.47, 17.77, 53.4,
         17.0, 52.63, 16.02, 51.76, 16.02, 50.78, 16.02, 49.58, 15.48, 48.6]], 'area': 59.2093, 'iscrowd': 0,
                                                                                          'image_id': 184613,
                                                                                          'bbox': [8.93, 48.6, 9.17,
                                                                                                   10.69],
                                                                                          'category_id': 1,
                                                                                          'id': 2022171}, {
                 'segmentation': [
                     [271.13, 109.9, 284.72, 110.6, 285.07, 110.6, 296.93, 103.63, 303.9, 95.96, 311.22, 93.86, 332.49,
                      91.08, 337.37, 92.12, 340.51, 93.17, 349.22, 90.73, 347.48, 89.33, 344.34, 87.59, 312.62, 89.68,
                      304.25, 89.33, 291.35, 89.68, 284.03, 95.26, 271.48, 108.86]], 'area': 557.6318999999996,
                 'iscrowd': 0, 'image_id': 184613, 'bbox': [271.13, 87.59, 78.09, 23.01], 'category_id': 21,
                 'id': 2069526}, {'segmentation': [
        [260.9, 80.2, 267.2, 81.36, 280.13, 93.3, 287.42, 89.98, 281.12, 96.11, 278.47, 101.09, 273.5, 107.55, 267.86,
         110.21, 257.75, 110.87, 259.74, 86.17]], 'area': 498.89059999999995, 'iscrowd': 0, 'image_id': 184613,
                                  'bbox': [257.75, 80.2, 29.67, 30.67], 'category_id': 21, 'id': 2069689}, {
                 'segmentation': [
                     [425.89, 143.13, 452.95, 135.27, 465.16, 132.65, 493.09, 168.44, 487.85, 186.76, 469.53, 212.95,
                      457.31, 233.89, 443.35, 281.02, 433.75, 276.65, 442.47, 239.13, 437.24, 243.49, 421.53, 262.69,
                      425.89, 244.36, 425.89, 244.36, 417.16, 240.87, 411.05, 243.49, 392.73, 270.55, 396.22, 290.62,
                      375.27, 317.67, 362.18, 311.56, 377.89, 274.04, 379.64, 262.69, 389.24, 247.85, 377.02, 247.85,
                      375.27, 263.56, 370.91, 277.53, 368.29, 295.85, 364.8, 301.09, 349.96, 301.09, 363.05, 264.44,
                      360.44, 241.75, 347.35, 244.36, 354.33, 261.82, 329.89, 256.58, 325.53, 251.35, 325.53, 251.35,
                      302.84, 266.18, 286.25, 269.67, 276.65, 278.4, 255.71, 268.8, 255.71, 259.2, 279.27, 240.0,
                      289.75, 211.2, 295.85, 205.09, 297.6, 191.13, 298.47, 185.89, 304.58, 205.09, 323.78, 203.35,
                      355.2, 193.75, 384.87, 164.95, 405.82, 152.73, 426.76, 148.36]], 'area': 18522.475850000006,
                 'iscrowd': 0, 'image_id': 184613, 'bbox': [255.71, 132.65, 237.38, 185.02], 'category_id': 21,
                 'id': 2177286}, {'segmentation': {
        'counts': [75, 14, 299, 7, 5, 1, 2, 2, 2, 8, 8, 3, 297, 33, 290, 47, 288, 49, 286, 51, 284, 53, 283, 54, 282,
                   55, 281, 30, 4, 22, 280, 20, 2, 5, 8, 25, 276, 19, 5, 1, 11, 29, 12, 2, 257, 18, 20, 32, 5, 4, 258,
                   14, 2, 1, 21, 40, 259, 12, 27, 38, 261, 10, 28, 37, 260, 12, 28, 36, 260, 15, 26, 35, 260, 17, 3, 1,
                   22, 33, 260, 21, 23, 16, 1, 15, 260, 20, 25, 13, 5, 13, 260, 17, 1, 2, 26, 10, 10, 10, 260, 15, 4, 1,
                   27, 7, 14, 8, 261, 11, 7, 1, 28, 2, 21, 5, 262, 8, 63, 3, 264, 3, 12099, 1, 334, 3, 332, 5, 330, 8,
                   328, 11, 325, 13, 323, 17, 2, 2, 315, 30, 306, 30, 306, 31, 305, 32, 304, 32, 304, 33, 303, 34, 302,
                   34, 302, 35, 301, 36, 300, 38, 298, 41, 295, 43, 294, 44, 292, 61, 276, 60, 22, 7, 247, 60, 22, 10,
                   244, 60, 22, 12, 242, 60, 22, 12, 242, 60, 22, 12, 242, 60, 22, 12, 242, 60, 22, 12, 242, 59, 24, 10,
                   242, 59, 26, 8, 242, 57, 31, 3, 244, 56, 279, 59, 277, 60, 276, 61, 275, 63, 273, 65, 271, 65, 271,
                   68, 268, 71, 265, 72, 264, 73, 262, 75, 259, 77, 258, 78, 257, 79, 256, 80, 255, 81, 255, 47, 2, 32,
                   255, 46, 4, 31, 255, 44, 8, 28, 256, 43, 12, 24, 257, 42, 17, 18, 259, 40, 297, 34, 303, 33, 305, 31,
                   51, 8, 251, 25, 51, 10, 250, 24, 51, 12, 249, 22, 52, 14, 248, 21, 53, 15, 247, 19, 55, 15, 247, 18,
                   56, 16, 1, 15, 230, 17, 57, 45, 217, 16, 58, 56, 206, 15, 59, 58, 204, 15, 59, 59, 203, 14, 61, 63,
                   198, 13, 62, 64, 197, 13, 63, 64, 197, 11, 64, 65, 197, 9, 64, 66, 199, 5, 65, 67, 268, 68, 267, 69,
                   267, 28, 2, 39, 266, 28, 16, 26, 265, 29, 26, 16, 265, 29, 27, 14, 266, 28, 29, 12, 267, 27, 31, 10,
                   268, 26, 33, 8, 269, 26, 35, 3, 272, 26, 310, 26, 310, 26, 311, 24, 313, 22, 316, 9, 2, 7, 30496, 6,
                   329, 8, 327, 10, 324, 13, 322, 14, 321, 16, 319, 18, 318, 18, 318, 18, 318, 18, 318, 18, 318, 18,
                   318, 19, 318, 19, 318, 22, 316, 21, 315, 22, 313, 26, 309, 28, 307, 30, 306, 31, 305, 31, 303, 34,
                   301, 35, 300, 37, 298, 38, 298, 38, 298, 38, 298, 38, 298, 38, 298, 38, 298, 38, 299, 25, 4, 8, 300,
                   10, 4, 2, 13, 7, 302, 7, 57817, 6, 329, 8, 326, 11, 323, 14, 319, 26, 308, 29, 306, 31, 304, 33, 302,
                   34, 301, 35, 301, 35, 300, 36, 299, 37, 298, 38, 298, 37, 299, 37, 299, 36, 300, 35, 301, 33, 303,
                   28, 308, 25, 312, 23, 314, 22, 316, 20, 316, 20, 315, 21, 315, 21, 315, 21, 315, 21, 315, 20, 316,
                   19, 317, 17, 320, 14, 323, 10, 328, 6, 6306], 'size': [336, 500]}, 'area': 8340, 'iscrowd': 1,
                                  'image_id': 184613, 'bbox': [0, 35, 481, 150], 'category_id': 1, 'id': 900100184613}]

dataDir = "J:/9.15/annotations_trainval2017/"
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
img_path = "J:/9.15/val2017/"
# imgcatIds = coco.getCatIds()[0]
# imgids = coco.getImgIds()
# imgid = coco.getImgIds(catIds=imgcatIds)[0]
# imgannids = coco.getAnnIds(imgid, imgcatIds)
# anns=coco.loadAnns(imgannids)

# front_path = img_path + coco.loadImgs(imgid)[0]['file_name']
# back_path = img_path + coco.loadImgs(imgids[random.randint(0,2)])[0]['file_name']

imgid = "0.jpg"
imgid1 = "1.jpg"
front_path = imgid
back_path = imgid1
# annids = coco.getAnnIds(724)
# anns = coco.loadAnns(annids)
# print(anns)
get_new_img(front_path, back_path, anns1_str)

#
# dataDir = "/run/media/kirin/DOC/ImageNet2012/coco2017/annotations_trainval2017/"
# dataType = 'val2017'
# annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
# coco = COCO(annFile)
# imgcatIds = coco.getCatIds()
# imgids=coco.getImgIds()
# imganns=coco.getAnnIds()
# print(len(imganns), len(imgids), len(imgcatIds))
