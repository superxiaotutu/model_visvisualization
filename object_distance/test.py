from pyecharts import Scatter

v1 = [10,10]
v2 = [60,20]
scatter = Scatter("散点图示例")
extra_name=['a','b','a','b','a','b']
scatter.add('b', v1, v2 ,is_datazoom_show =True,is_legend_show=False,mark_point=['max', 'min'])
scatter.add('a', v1, v2 ,is_datazoom_show =True,is_legend_show=False,mark_point=['max', 'min'])

scatter.render()