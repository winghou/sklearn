import matplotlib.pyplot as plt

plt.plot([3,1,3,4,2])
plt.ylabel("grade")
plt.savefig('绘图1',dpi=600)  #输出图形存储为文件，默认PNG格式，通过dpi修改输出质量
plt.show()