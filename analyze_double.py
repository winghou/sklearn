#!/usr/bin/python
# -*- coding:UTF-8 -*-
# Description:线性回归是利用数理统计中回归分析，来确定两种或两种以上变量间相互依赖的定量关系的一种统计方法。
#              线性回归利用称为线性回归方程的最小平方函数对一个或多个自变量和因变量之间关系进行建模。
#线性回归的实际用途：如果目标是预测或映射，线性回归可以用来对观测数据集的y和X的值拟合出一个预测模型。
#                   当完成这样一个模型以后，对于一个新增的X值，在没有给定与它相配对的y的情况下，可以用这个
#                    拟合过的模型预测出一个y值
#技术路线：sklearn.linear_model.LinearRegression
#可行性分析：

#导入需要的包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

#读取文件
desktop_path = 'E:/GitHub/Lottery/'

df = pd.read_csv(desktop_path+'hun.txt',header=None,sep=',')

#读取日期
tdate = sorted(df.loc[:,0])
print(tdate)

#将以列项为数据，将球号码取出，写入到csv文件中，并取50行数据
# Function to red number to csv file
def ReadToCsv(h_num,num,csv_name):
    h_num = df.loc[:,num:num].values
    h_num = h_num[50::-1]
    renum2 = pd.DataFrame(h_num)
    renum2.to_csv(csv_name,header=None)

    fp = open(csv_name,'r')  #读取
    s = fp.read()
    fp.close()
    a = s.split('\n')
    a.insert(0, 'numid,number')  #插入索引，数字
    s = '\n'.join(a)

    fp = open(desktop_path+csv_name, 'w')  #写入
    fp.write(s)
    fp.close()

#调用取号码函数
# create file 把数据存储成一个.csv文件
ReadToCsv('red1',1,'rednum1data.csv')
ReadToCsv('red2',2,'rednum2data.csv')
ReadToCsv('red3',3,'rednum3data.csv')
ReadToCsv('red4',4,'rednum4data.csv')
ReadToCsv('red5',5,'rednum5data.csv')
ReadToCsv('red6',6,'rednum6data.csv')
ReadToCsv('blue1',7,'bluenumdata.csv')



#获取数据，X_parameter为numid数据,Y_parameter为number数据
# Function to get data
def get_data(file_name):
    data = pd.read_csv(file_name)  #将.csv数据读入Pandas数据帧
    X_parameter = [] #用来存储数据中的numid数据
    Y_parameter = [] #用来存储数据中的number数据
    for numid,number in zip(data['numid'],data['number']): #逐行进行操作，循环遍历所有数据
        X_parameter.append([float(numid)])  #将读取的数据转换为int型，并分别写入
        Y_parameter.append(float(number))

    length = len(X_parameter)  #求得X_parameter的长度，即数据的总数
    print(length)
    X_parameter = np.array(X_parameter).reshape([length,1]) #将X_parameter转化为数组，并变为2维，以符合线性回归你和函数输入参数要求
    Y_parameter = np.array(Y_parameter)
    return X_parameter,Y_parameter  #把Pandas数据帧转换为X_parameter和Y_parameter数据，并返回他们
   # print(X_parameter,Y_parameter)

#训练线性模型
# Function for Fitting our data to Linear model
def linear_model_main(X_parameters,Y_parameters,predict_value):  #模板
    # Create linear regression object
    linear = linear_model.LinearRegression()  #调用线性回归模块，建立回归方程，拟合数据。
    # regr = LogisticRegression()
    linear.fit(X_parameters, Y_parameters,sample_weight=None)  #拟合输入输出数据，X为训练向量，y为相对于X的目标向量
    predict_outcome = linear.predict(predict_value)  #模板
    predictions = {}  #创建一个名称为predictions的字典，存着θ0、θ1和预测值，并返回predictions字典为输出。
    predictions['intercept'] = linear.intercept_  #回归方程截距
    predictions['coefficient'] = linear.coef_  #回归方程系数
    predictions['predicted_value'] = predict_outcome  #模板
    return predictions



#获取预测结果函数
def get_predicted_num(inputfile,num):
    X,Y = get_data(inputfile)
    predictvalue = 51 #要预测的是第51期
    predictvalue = np.array(predictvalue, dtype=np.int32).reshape(1, -1)  #新版必须是2维哦
    result = linear_model_main(X,Y,predictvalue)
    print("num "+ str(num) +" Intercept value " , result['intercept'])  #截距值就是θ0的值
    print("num "+ str(num) +" coefficient" , result['coefficient'])  #系数就是θ1的值
    print("num "+ str(num) +" Predicted value: ",result['predicted_value'])


#调用函数分别预测红球、蓝球
get_predicted_num('rednum1data.csv',1)
get_predicted_num('rednum2data.csv',2)
get_predicted_num('rednum3data.csv',3)
get_predicted_num('rednum4data.csv',4)
get_predicted_num('rednum5data.csv',5)
get_predicted_num('rednum6data.csv',6)
get_predicted_num('bluenumdata.csv',1)

"""
#获取X,Y数据预测结果
X,Y = get_data('rednum1data.csv')
predictvalue = 51
predictvalue = np.array(predictvalue, dtype=np.int32).reshape(1, -1)  #新版必须是2维哦
result = linear_model_main(X,Y,predictvalue)
print ("red num 1 Intercept value " , result['intercept'])
print ("red num 1 coefficient" , result['coefficient'])
print ("red num 1 Predicted value: ",result['predicted_value'])
"""

# Function to show the resutls of linear fit model  显示出数据拟合的直线
def show_linear_line(X_parameters,Y_parameters):
    # Create linear regression object
    linear = linear_model.LinearRegression() #调用线性回归模块，建立回归方程，拟合数据。
    #regr = LogisticRegression()
    linear.fit(X_parameters, Y_parameters) #拟合输入输出数据，X为训练向量，y为相对于X的目标向量
    plt.figure(figsize=(12,6),dpi=80)
    plt.legend(loc='best')
    plt.scatter(X_parameters,Y_parameters,color='red') #绘制数据散点
    plt.plot(X_parameters,linear.predict(X_parameters),color='blue',linewidth=4) #绘制回归线
    plt.xticks(())
    plt.yticks(())
    plt.show()

#显示模型图像，如果需要画图，将“获取X,Y数据预测结果”这块注释去掉，“调用函数分别预测红球、蓝球”这块代码注释下
#show_linear_line(X,Y)

