import numpy as np
import matplotlib.pyplot as plt
plt.subplots_adjust(hspace=0.3, wspace=0.3)
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.title('Subplot 1: sin(x)')
plt.show()