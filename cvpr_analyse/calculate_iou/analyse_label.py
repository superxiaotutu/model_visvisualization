import json

import numpy
import matplotlib.pyplot as plt
arr=numpy.zeros((1000,1000))
with open('imagenet.json') as f:
    data=json.load(f)
plt.imshow(plt.imread('adv.png'))
plt.show()
# with open('7rar_groundtruth_iou.txt') as f:
#     lines=f.readlines()
#     for l in lines:
#         l=l.strip('\n').split()
#         arr[int(l[0])][int(l[1])]+=1
#
#     key=(numpy.argmax(arr,axis=1))
#
#     value=(numpy.max(arr,axis=1))
# # 最大值的原标签
# # print(numpy.argmax(value))
# # 最大值的被进攻后的标签
# # print(key[numpy.argmax(value)])
#
# arr=numpy.zeros((1000,1000))
# for i,v in enumerate(key):
#     arr[i][v]=value[i]
#     # plt.bar(i,value[i])
# # plt.show()
# for i in range(400,500):
#     print(data[i],data[numpy.argmax(arr[i])])
#     print([i],numpy.argmax(arr[i]))
#     # plt.plot(arr)
#     # plt.show()
arr=numpy.zeros((1000))
with open('7rar_groundtruth_iou.txt') as f:
    lines=f.readlines()
    for l in lines:
        l=l.strip('\n').split()
        if l[0]==l[1]:
            arr[int(l[0])]+=1

    key=(numpy.argmax(arr))

    value=(numpy.max(arr))
#防御成功top5
top5=numpy.argsort(arr)[-5:]
bot5=numpy.argsort(arr)[:5]
print(numpy.sort(arr)[-5:],top5)
print(numpy.sort(arr)[:5],numpy.argsort(arr)[:5])

for i in top5:
    print(data[int(i)])
print()

for i in bot5:
    print(data[int(i)])