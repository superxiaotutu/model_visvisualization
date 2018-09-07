from pyecharts import Scatter
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
print(np.arange(1, 1001, 1))
a=sns.barplot(y=['tench,-Tinca-tinca'], x=[1], orient='h')
plt.show()
print(a)