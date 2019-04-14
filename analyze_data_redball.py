#!/usr/bin/python
# -*- coding:UTF-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# 读取文件
desktop_path = 'E:/GitHub/Lottery/'

df = pd.read_csv(desktop_path+'hun.txt', header=None, sep=',''')  #header 用作列名的行号。sep 对行中各字段进行拆分的正则
#print (df)
#print (df[1:3])  #第2到第3行（索引0开始为第一行，1代表第二行，不包含第四行）
#print (df.loc[0:10,:]) #第1行到第9行的全部列
print (df.loc[:,1:6])   #全部行的第1到第6列  6个红球
#tdate = sorted(df.loc[:, 0]) # 仅对第1列数据（日期）升序
#print  (tdate)
h1 = df.loc[:, 1] #取第1列的值
print  (h1)
h2 = df.loc[:, 2] #取第2列的值
h3 = df.loc[:, 3] #取第3列的值
h4 = df.loc[:, 4] #取第4列的值
h5 = df.loc[:, 5] #取第5列的值
h6 = df.loc[:, 6] #取第6列的值

# 将数据合并到一起
all = h1.append(h2).append(h3).append(h4).append(h5).append(h6)
alldata = list(all)
#print(alldata)
print(len(alldata))


fenzu = pd.value_counts(all, ascending=False)
print(fenzu)

x = list(fenzu.index[:])
y = list(fenzu.values[:])
print(x)
print(y)

# print type(fenzu)
plt.figure(figsize=(10, 6), dpi=70)
plt.legend(loc='best', )

# plt.plot(fenzu,color='red')
plt.bar(x, y, alpha=.5, color='r', width=0.8)
plt.title('The red ball number')
plt.xlabel('red number')
plt.ylabel('times')
plt.grid(True)


# 循环，为每个柱形添加文本标注
# 居中对齐
for xx, yy in zip(x,y):
    plt.text(xx, yy+0.5, str(yy), ha='center')

plt.show()

