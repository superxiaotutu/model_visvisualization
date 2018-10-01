import pandas
import matplotlib.pyplot as plt
import numpy as np

data = pandas.read_csv('out.csv', header=None, sep=',')
count = int(data.count()[0])
x_arr = []
y_arr = []
x_arr1 = []
y_arr1 = []
count_y = 0
count_n = 0
plt.figure(111, (1000, 1000))
#取每列数据
true_matrix = np.asanyarray(data.loc[:, 1])
ori_matrix = np.asanyarray(data.loc[:, 2])
adv_matrix = np.asanyarray(data.loc[:, 3])
yes_matrix = np.asanyarray(data.loc[:, 5])
#计算x，y矩阵
x_matrix = true_matrix - ori_matrix
y_matrix = ori_matrix - adv_matrix
x_matrix = np.clip(x_matrix, 0, None)
y_matrix = np.clip(y_matrix, 0, None)

new_yes_matrix = []
new_x_matrix = []
new_y_matrix = []
print(ori_matrix.shape)
#新建矩阵，把小于等于0的值抛弃
for index, value in enumerate(yes_matrix):
    if x_matrix[index] > 0 and y_matrix[index] > 0:
        new_yes_matrix.append(value)
        new_x_matrix.append(x_matrix[index])
        new_y_matrix.append(y_matrix[index])
    else:
        continue
# x,y归一化计算
x_matrix_std = np.std(new_x_matrix)
x_matrix_mean = np.mean(new_x_matrix)
y_matrix_std = np.std(new_y_matrix)
y_matrix_mean = np.mean(new_y_matrix)


new_x_matrix = (new_x_matrix - x_matrix_mean) / (x_matrix_std)
new_y_matrix = (new_y_matrix - y_matrix_mean) / (y_matrix_std)
#如果成功防御则加入x_arr中，否则加入x_arr1
for index, value in enumerate(new_yes_matrix):
    if value == 'Yes':
        x_arr.append(new_x_matrix[index])
        y_arr.append(new_y_matrix[index])
    else:
        x_arr1.append(new_x_matrix[index])
        y_arr1.append(new_y_matrix[index])

print(len(x_arr), len(x_arr1))
plt.scatter(x_arr1, y_arr1, color='r', alpha=0.3)
plt.scatter(x_arr, y_arr, color='b', alpha=0.3)
plt.show()
# for i in range(count):
#     record = data.iloc[[i]]
#     true = record[1]
#     ori = record[2]
#     adv = record[3]
#     yes = record[5].iloc[0]
#     x = float((true - ori) / (1 - ori))
#     y = float((ori - adv) / (ori))
#     if x < 0 or y < 0:
#         continue
#     if str(yes) == 'Yes':
#         count_y += 1
#         x_arr.append(x)
#         y_arr.append(y)
#     else:
#         count_n += 1
#         x_arr1.append(x)
#         y_arr1.append(y)
# print(count_y, count_n)
