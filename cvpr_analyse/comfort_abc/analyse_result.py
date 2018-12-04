import xlrd
import xlutils.copy
import xlwt
import xlsxwriter
workbook = xlwt.Workbook()
worksheet=workbook.add_sheet('0')

with open('result.txt','r') as f:
    lines=f.readlines()
    for i,l in enumerate(lines):
        l=(l.strip('\n').split(' '))
        print(l)
        worksheet.write(i,0, l[0])
        worksheet.write(i,1, l[1])
        worksheet.write(i,2, l[2])
        worksheet.write(i,3, l[3])
        worksheet.write(i,4, l[4])
        worksheet.write(i,5, l[5])
        worksheet.write(i,6, l[6])

        workbook.save('a.xls')
