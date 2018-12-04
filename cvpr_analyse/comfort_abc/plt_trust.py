import pandas
import matplotlib.pyplot as plt
import numpy as np
x_arr=[]
y_arr=[]
labels=[]
x_true_arr=[]
y_true_arr=[]
y_false_arr_f=[]
y_true_arr_f=[]
x_false_arr_f=[]
x_true_arr_f=[]
name='a'
with open(name,'r') as f:
    lines=f.readlines()
    for i,l in enumerate(lines):
        l_arr=l.split(' ')
        is_attack=l_arr[0]
        trust=round(float(l_arr[3]),5)
        if is_attack == 'defense':

            y_true_arr_f.append(trust)
            x_true_arr_f.append(i)
        else:
            y_false_arr_f.append(trust)
            x_false_arr_f.append(i)
plt.figure(figsize=(20, 15))
my_y_ticks = np.arange(0, 1, 0.05)
my_x_ticks = np.arange(0, len(lines), 5000)
plt.xticks(my_x_ticks)

plt.yticks(my_y_ticks)

plt.scatter(x_false_arr_f, y_false_arr_f, color='r',alpha=0.5)

plt.scatter(x_true_arr_f, y_true_arr_f, color='b',alpha=0.3)

plt.show()
