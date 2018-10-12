fp = open('ILSVRC2012_val_00000293.xml')
from  matplotlib import  pyplot as plt
import numpy as np
def get_gound_truth():
    ground_truth=np.zeros((299,299,1))
    for p in fp:
        if '<size>' in p :
            width=int(next(fp).split('>')[1].split('<')[0])
            height=int(next(fp).split('>')[1].split('<')[0])

        if '<object>' in p:
            print(next(fp).split('>')[1].split('<')[0])
        if '<bndbox>' in p:
            xmin=int(next(fp).split('>')[1].split('<')[0])
            ymin=int(next(fp).split('>')[1].split('<')[0])
            xmax=int(next(fp).split('>')[1].split('<')[0])
            ymax=int(next(fp).split('>')[1].split('<')[0])
            return int(xmin/width*299),int(ymin/height*299),int(xmax/height*299),int(ymax/height*299)
print(get_gound_truth())

plt.imshow()