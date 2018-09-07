import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 第一个图，我们来探索下英雄与坏蛋们眼睛颜色的分布
f, ax = plt.subplots(figsize=(12, 20))
data = np.load('feature_diff.npy')

with open('V3_namelist.txt', 'r') as f_r:
    extra_names = f_r.readlines()
    extra_names = extra_names
    for i,v in enumerate(extra_names):
        extra_names[i]=extra_names[i].strip('\n')
        extra_names[i]=extra_names[i].replace(' ','-')

    print(extra_names)
    # sns.barplot(y=[i for i in range(100)], x=data[:100], orient='h')
    sns.barplot(y=extra_names, x=data, orient='h')
    plt.xticks(rotation='horizontal')
    plt.show()
