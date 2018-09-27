import pandas
from pyecharts import Scatter

data=pandas.read_csv('out.csv',header=None,sep=',')
count=int(data.count()[0])
bar = Scatter('scatter', 'scatter', width=2100, height=1200, )
x_arr=[]
y_arr=[]
for i in range(count) :
    record = data.iloc[[i]]
    true=record[1]
    ori=record[2]
    adv=record[3]
    x_arr.append(true-ori)
    y_arr.append(ori-adv)
bar.add('scatter',x_arr,y_arr)
bar.render('loss.html')
