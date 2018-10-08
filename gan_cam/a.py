import numpy as np

a=np.asarray(
    [[0,0,0,0,0,0,0,0],
     [1,1,1,1,1,0,0,1]]
)
b=np.asarray(
    [[0,0,0,0,0,0,0,0],
     [1,1,1,1,1,0,0,1]]
)
count=0
c=a+b


print(c[c!=0].size)
iou=c[c == 2].size/c[c!=0].size
print(iou)
