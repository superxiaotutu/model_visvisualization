import pyecharts
from pyecharts import Scatter
import numpy as np

model_name = "deepfool"
data = np.load(model_name + '_grad_data_all.npy')
dirname = 'output'
bar_name = "图"
desc_name = ""
desc_coor_y = "均值"
x_arr=[]
y_arr=[]
for x_index in range(data.shape[2]):
    for index in range(data.shape[0]):
        coor_x = list(data[index][0][x_index])
        coor_y = list(data[index][1][x_index])
        x_arr+=coor_x
        y_arr+=coor_y

    # for x_index in range(data.shape[2]):
    #     with open('convname.txt', 'r') as f:
    #         desc_name = f.readlines()[index]
    #     coor_x = data[index][0][x_index]
    #     coor_y = data[index][1][x_index]
    # v1 = [2.0, 4.9, 7.0, 23.2, 25.6, 76.7, 135.6, 162.2, 32.6, 20.0, 6.4, 3.3]
    # v2 = [2.6, 5.9, 9.0, 26.4, 28.7, 70.7, 175.6, 182.2, 48.7, 18.8, 6.0, 2.3]
    bar = Scatter(bar_name, desc_name)
    bar.add(desc_coor_y, x_arr/data.shape[0], y_arr/data.shape[0], mark_line=["average"], mark_point=["max", "min"])
    # bar.add("evaporation", attr, v2, mark_line=["average"], mark_point=["max", "min"])
    bar.render(path=dirname + '/' + str(1) + '.html')
    with open(model_name + '_results.html', 'a', encoding='utf8') as f:
        with open(dirname + '/' + str(1) + '.html', 'r', encoding='utf8') as f_r:
            f.write(f_r.read())
