import xlrd
import matplotlib.pyplot as plt

workbook = xlrd.open_workbook(r'cifar_ab.xlsx')
sheet = workbook.sheet_by_index(0)
offset = 7

c = sheet.col_values(2)[1:offset]
a_correct = sheet.col_values(3)[1:offset]
b_correct = sheet.col_values(4)[1:offset]
c_correct = sheet.col_values(6)[1:offset]
print((c), a_correct, b_correct, c_correct)
plt.title('a=8 b=4 Tuning Results')
plt.plot(c, a_correct, color='green', label='rar accuracy')
plt.plot(c, b_correct, color='red', label='stepll accuracy')
plt.plot(c, c_correct, color='skyblue', label='double distance')
plt.legend()

plt.xlabel('c')
plt.ylabel('rate')
plt.show()
