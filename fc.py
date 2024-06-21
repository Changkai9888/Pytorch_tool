import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from datetime import datetime
import matplotlib.dates as mdate
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.family'] = 'Arial'
###########
def plot(data,log=0,label=["y"],k=0):
    """
    绘制折线图的函数。

    参数:
    data (list): 要绘制的数据列表。
    """
    # 检查输入是否为列表
    #if not isinstance(data, list):
        #raise ValueError("输入必须是列表类型")
    # 绘制折线图
    if k==1:#转置处理
        data=np.array(data).T
    plt.figure(figsize=(10, 5))  # 设置图形大小
    if len(np.array(data).shape)==1:
        if log==1:
            data=np.log(data)
        plt.plot(data, marker='')  # 绘制折线图，并在每个数据点上标记一个圆点
    else:
        for i in range(len(data)):
            data_i=data[i] if log==0 else np.log(data[i])
            plt.plot(data_i, marker='', label=label[i] if len(label)!=1 else None)  # 绘制折线图，并在每个数据点上标记一个圆点
    #plt.plot(data, marker='')  # 绘制折线图，并在每个数据点上标记一个圆点
    plt.legend() if len(label)!=1 else None
    plt.title('Line Plot')  # 设置图形标题
    plt.xlabel('Index')  # 设置x轴标签
    plt.ylabel('Value')  # 设置y轴标签
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图形
def plot_timex(time,data, label="y", title="",log=0):#x坐标为时间的折线图
    time_x=[datetime.strptime(d, "%Y%m%d").date() for d in time]
    ax=plt.subplot(111)
    ax.xaxis.set_major_formatter(mdate.DateFormatter("%Y%m%d"))
    plt.xticks(pd.date_range(time_x[0],time_x[-1],freq='D'),rotation=45)
    if len(np.array(data).shape)==1:
        if log==1:
            data=np.log(data)
        ax.plot(time_x,data,marker='',label=label)
    else:
        for i in range(len(data)):
            data_i=data[i] if log==0 else np.log(data[i])
            ax.plot(time_x,data_i,marker='',label=label[i])
    plt.legend()
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.set_title(title)
    plt.show()
#####
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
