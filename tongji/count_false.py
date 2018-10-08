import pandas
import matplotlib.pyplot as plt
import numpy as np
x_arr=[]
y_arr=[]
labels=[]
x_true_arr=[]
y_true_arr=[]
x_false_arr=[]
y_false_arr=[]
name='adv_mutilayer_data_analyse_l1'
with open('input/'+name+'.txt','r') as f:
    lines=f.readlines()
    for l in lines:
        l_arr=l.split(',')
        l_arr[:17]=list(l_arr[:17])
        for i,j in enumerate(l_arr[:17]):
            l_arr[:17][i]=(float('%.1f' % float(l_arr[:17][i])))
        labels.append(l_arr[17])
        x_arr.append(l_arr[:17])
        # l_arr[0]=l_arr[0].strip(' ')
        # l_arr[1]=l_arr[1].strip(' ')
        # x_arr.append(float('%.15f' % float(l_arr[0])))
        # labels.append(l_arr[1])
        # y_arr.append(int(l_arr[2]))
for i,j in enumerate(labels):
    if j == ' True':
        x_true_arr.append(x_arr[i])
    else:
        x_false_arr.append(x_arr[i])
plt.figure(name, (1000, 1000))
x_false_arr_f=[]
y_false_arr_f=[]
x_true_arr_f=[]
y_true_arr_f=[]
for i in range(len(x_false_arr)):
    x_false_arr_f.append(x_false_arr[i])
    y_false_arr_f.append(np.arange(17))

for i in range(len(x_true_arr)):
    x_true_arr_f.append(x_true_arr[i])
    y_true_arr_f.append(np.arange(17))
plt.scatter(x_false_arr_f, y_false_arr_f, color='r', alpha=0.1)

plt.scatter(x_true_arr_f, y_true_arr_f, color='b', alpha=0.05)

plt.show()




# data = pandas.read_csv('out.csv', header=None, sep=',')
# # count = int(data.count()[0])
# x_arr = []
# y_arr = []
# x_arr1 = []
# y_arr1 = []
# count_y = 0
# count_n = 0
# plt.figure(111, (500, 500))
# label=np.asanyarray(data.loc[:, 4])
# count=np.max(label)+1
# print(count)
# label_matrix = np.zeros(count)
# label_matrix_x = np.arange(count)
#
#
# # 取每列数据
# yes_matrix = np.asanyarray(data.loc[:, 5])
# for i, v in enumerate(yes_matrix):
#     if v == 'No':
#         label_matrix[label[i]] += 1
# sorted_matrix=np.argsort(label_matrix)
# #最能防御的
# front_labels=sorted_matrix[:20]
# #最不能防御的
# end_labels=sorted_matrix[-20:]
# with open('imagenet_labels','r',encoding='utf8') as f:
#     ls=f.readlines()
#     for i in end_labels:
#         print(ls[i])
#
# # plt.bar(label_matrix_x,label_matrix)
# # plt.show()