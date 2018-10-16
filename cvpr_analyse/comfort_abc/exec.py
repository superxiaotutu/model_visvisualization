import sys
import os

# a_list,b_list,c_list=[8],[2,2.5,3,3.5,4,4.5,5],[1]
#
# a_list,b_list,c_list=[8],[6,6.5,5.5],[1]
# a_list,b_list,c_list=[11,10,9],[4],[1]
#
# a_list,b_list,c_list=[5,4,3,2,1,8],[4],[1]
#
# a_list,b_list,c_list=[8],[4],[0.2,0.4]
#
# a_list,b_list,c_list=[8],[4],[0.8,1,1.2]
a_list,b_list,c_list=[8],[4],[1.4,1.6,1.8,2,0.6]


for ai, a in enumerate(a_list):
    for bi, b in enumerate(b_list):
        for ci, c in enumerate(c_list):
            # result_file = 'result' + str(a) + '_' + str(b) + '_' + str(c) + '.txt'
            # if os.path.isfile(result_file):
                s=str(a)+"-"+str(b)+"-"+str(c) +"-2"
                s="nohup python3 cifar10_arg.py "+s +" &"
                print(s)
                os.system(s)
# for i in enumerate