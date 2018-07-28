import pyecharts
from pyecharts import Bar

import numpy as np

data = np.load('sum.npy')
dirname = 'output'
bar_name = "图"
desc_name = ""
desc_coor_y = "均值"
for index in range(data.shape[0]):
    with open('convname', 'r') as f:
        desc_name=f.readlines()[index]
    coor_x = [i for i in range((data[index].shape)[0])]
    coor_y = data[index]
    # v1 = [2.0, 4.9, 7.0, 23.2, 25.6, 76.7, 135.6, 162.2, 32.6, 20.0, 6.4, 3.3]
    # v2 = [2.6, 5.9, 9.0, 26.4, 28.7, 70.7, 175.6, 182.2, 48.7, 18.8, 6.0, 2.3]
    bar = Bar(bar_name, desc_name)
    bar.add(desc_coor_y, coor_x, coor_y, mark_line=["average"], mark_point=["max", "min"])
    # bar.add("evaporation", attr, v2, mark_line=["average"], mark_point=["max", "min"])
    bar.render(path=dirname + '/' +str(index) + '.html')
    with open('results.html', 'a', encoding='utf8') as f:
        with open(dirname + '/' + str(index) + '.html', 'r', encoding='utf8') as f_r:
            f.write(f_r.read())
