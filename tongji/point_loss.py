import pandas
import matplotlib.pyplot as plt
data = pandas.read_csv('out.csv', header=None, sep=',')
count = int(data.count()[0])
x_arr = []
y_arr = []
x_arr1 = []
y_arr1 = []
count_y=0
count_x=0
plt.figure(111,(1000,1000))
for i in range(100):
    record = data.iloc[[i]]
    true = record[1]
    ori = record[2]
    adv = record[3]
    yes = record[5].iloc[0]
    if str(yes) == 'Yes':
        count_y+=1
        x_arr.append(float(true - ori))
        y_arr.append(float(ori - adv))
    else:
        count_x+=1
        x_arr1.append(float(true - ori))
        y_arr1.append(float(ori - adv))
print(count_y,count_x)
plt.scatter(x_arr1,y_arr1,color='r',alpha=0.4)
plt.scatter(x_arr,y_arr,color='b',alpha=0.4)
plt.show()