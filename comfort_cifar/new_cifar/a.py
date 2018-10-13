import sys
import os
# a_list,b_list,c_list=[15,14],[4,5],[1]
a_list,b_list,c_list=[8,7],[4,5],[1,2]
# a_list,b_list,c_list=[10,7],[3,5],[1]

# a_list,b_list,c_list=[6,7],[3,2],[1,2]
# a_list,b_list,c_list=[20,10],[5,8],[1,2]
# a_list,b_list,c_list=[16,17],[5,6],[3,2]
# a_list,b_list,c_list=[20,15],[6,7],[1,2]
# a_list,b_list,c_list=[14,15],[7,6],[3,2]
# a_list,b_list,c_list=[14,15],[11,8],[1,2]
# a_list,b_list,c_list=[16,17],[9,10],[1,2]
for ai, a in enumerate(a_list):
    for bi, b in enumerate(b_list):
        for ci, c in enumerate(c_list):
            s=str(a)+"-"+str(b)+"-"+str(c)
            s="nohup python cifar10_arg.py "+s +" &"
            print(s)
            os.system(s)
# for i in enumerate