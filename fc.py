import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from datetime import datetime
import matplotlib.dates as mdate
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
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
#交易结果画图
def plot_trade(close,pos,right):
    a,b,c=(i.detach().numpy() if str(type(i))=="<class 'torch.Tensor'>" else i for i in (close,pos,right))
    # 检验数据
    if not(len(a)==len(b)==len(c)):
        raise ValueError("输入序列长度不等！")
    lon=len(a)
    x = list(range(lon))
    # 创建共享坐标轴的子图
    fig, axs = plt.subplots(3, 1, figsize=(9, 6),  height_ratios=[3, 1, 1],
                            sharex=True, sharey=False)  # 关键参数[1,7](@ref)
    plt.subplots_adjust(hspace=0.05, right=0.9)  # 控制子图间距[6](@ref)
    # 绘制每个子图
    axs[0].plot(x, a, 'r', label='close')
    #axs[0].set_ylim(min_val - margin, max_val + margin)
    #axs[0].legend()#loc='upper left')               # 去掉图例边框（可选）[7](@ref))
    axs[0].legend(loc='upper left',bbox_to_anchor=(1.0, 1),borderaxespad=-0.5)  # 右侧外定位[4,7](@ref)) 
    axs[0].set_title('交易结果', y=1.1)  # 标题上移

    axs[1].plot(x, [0]*len(x),'k--',linewidth=0.5);
    axs[1].plot(x, [1]*len(x),'k--',linewidth=0.5)
    axs[1].plot(x, [-1]*len(x),'k--',linewidth=0.5)
    axs[1].plot(x, b, 'g', label='pos') 
    axs[1].set_ylim(-1.2, 1.2)
    axs[1].legend(loc='upper left',bbox_to_anchor=(1.0, 1),borderaxespad=-0.5)#loc='upper left')
    axs[1].tick_params(labelbottom=False)  # 隐藏中间图的x轴标签[6](@ref)

    axs[2].plot(x, c, 'b', label='right')
    axs[2].legend(loc='upper left',bbox_to_anchor=(1.0, 1),borderaxespad=-0.5)#loc='upper left')
    axs[2].set_xlabel('索引')  # 仅底部显示x轴标签[1](@ref)

    # 同步缩放设置
    for ax in axs:
        ax.label_outer()  # 隐藏内部冗余标签[7](@ref)
        ax.grid(alpha=0.3)  # 添加辅助网格
    plt.show()
####
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
