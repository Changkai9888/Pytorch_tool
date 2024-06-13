import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from datetime import datetime
import matplotlib.dates as mdate
plt.rcParams['font.sans-serif']=['SimHei']
###########
def plot(data):
    """
    绘制折线图的函数。

    参数:
    data (list): 要绘制的数据列表。
    """
    # 检查输入是否为列表
    if not isinstance(data, list):
        raise ValueError("输入必须是列表类型")

    # 绘制折线图
    plt.figure(figsize=(10, 5))  # 设置图形大小
    plt.plot(data, marker='')  # 绘制折线图，并在每个数据点上标记一个圆点
    plt.title('Line Plot')  # 设置图形标题
    plt.xlabel('Index')  # 设置x轴标签
    plt.ylabel('Value')  # 设置y轴标签
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图形
def plot_timex(time,data):#x坐标为时间的折线图
    time_x=[datetime.strptime(d, "%Y%m%d").date() for d in time]
    ax=plt.subplot(111)
    ax.xaxis.set_major_formatter(mdate.DateFormatter("%Y年%m月%d日"))
    plt.xticks(pd.date_range(time_x[0],time_x[-1],freq='D'),rotation=45)
    ax.plot(time_x,data,marker='')
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.show()
