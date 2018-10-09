count=0
b_count=0
sum=0
b_sum=0
with open('grad_result_fgsm_final.txt','r') as  f:
    lines=f.readlines()
    for l in lines:
        arr=l.split(' ')
        iou=arr[7]
        if arr[6]==arr[5]:
            sum += float(iou)
            count+=1
        else:
            b_sum+=float(iou)
            b_count+=1
    print(sum/count,b_sum/b_count)
