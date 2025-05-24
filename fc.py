import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from datetime import datetime, timedelta
import matplotlib.dates as mdate
import gzip,shutil,hashlib,time,pickle,os
from functools import wraps
import inspect
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
#装饰器：函数的计时功能。
def timer(func):
    """装饰器：计算函数运行时间"""
    @wraps(func)  # 保留原函数元信息（如函数名、文档说明）[4,7](@ref)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 使用高精度计时器（推荐）[6](@ref)
        result = func(*args, **kwargs)    # 执行原函数
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"函数 {func.__name__} 运行耗时: {duration:.6f} 秒")  # 输出微秒级精度[6](@ref)
        return result
    return wrapper
#装饰器：装饰器：加快反复调用 很耗时的 相同函数 相同参数 时的速度。
def disk_cache(func):
    '''装饰器：加快反复调用 很耗时的 相同函数 相同参数 时的速度。
    相同的 函数 传递相同的 参数 时候，第一次调用计算结果保存到硬盘，第二次则不计算直接调用保存的结果进行输出。
    保存在"./function_cache"，每次会比对函数的代码内容是否变化？参数传递是否变化？
    每日首次运行时，自动检测，如果保存的结果在近3日没有被调用，则自动删除，防止硬盘空间不足。
    对大文件进行压缩。'''
    """硬盘缓存装饰器（集成GZIP压缩+函数指纹+参数指纹+过期清理）"""
    CACHE_ROOT = "./function_cache"
    CACHE_DAYS = 3
    def cleanup_old_cache():
        """清理过期缓存（基于网页9的目录时间戳比对）"""
        now = datetime.now()
        for func_dir in os.listdir(CACHE_ROOT):
            dir_path = os.path.join(CACHE_ROOT, func_dir)
            timestamp_file = os.path.join(dir_path, "timestamp.txt")
            if os.path.exists(timestamp_file):
                with open(timestamp_file, 'r') as f:
                    cache_time = datetime.fromisoformat(f.read())
                if abs((now - cache_time).days) > CACHE_DAYS:
                    # 删除整个过期目录（网页10的shutil.rmtree方法）
                    shutil.rmtree(dir_path)
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 1. 生成函数版本指纹（网页5的SHA256哈希方法）
        func_code = inspect.getsource(func).encode()
        func_hash = hashlib.sha256(func_code).hexdigest()[:16]
        
        # 2. 生成参数指纹（网页6的pickle序列化+MD5哈希）
        param_data = pickle.dumps((args, kwargs))
        param_hash = hashlib.md5(param_data).hexdigest()
        
        # 3. 创建缓存目录
        cache_dir = os.path.join(CACHE_ROOT, f"func_{func_hash}")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 4. 缓存文件路径（保持.pkl后缀但实际为gzip压缩文件）
        cache_file = os.path.join(cache_dir, f"param_{param_hash}.pkl")
        timestamp_file = os.path.join(cache_dir, "timestamp.txt")
        
        # 5. 检查缓存有效性（网页7的流式压缩读法）
        if os.path.exists(cache_file):
            # 检查时间戳是否在3天内（网页9的时间比对逻辑）
            with open(timestamp_file, 'r') as f:
                cache_time = datetime.fromisoformat(f.read())
            with open(timestamp_file, 'w') as f:
                f.write(datetime.now().isoformat())#更新时间戳
            if datetime.now() - cache_time >= timedelta(days=CACHE_DAYS):
                # 运行过期清理
                cleanup_old_cache()
            # 使用gzip流式解压读取（网页3的GzipFile方法）
            with gzip.open(cache_file, 'rb') as f:
                return pickle.load(f)
        # 6. 执行计算并保存结果（网页7的流式压缩写法）
        result = func(*args, **kwargs)
        # 使用gzip压缩序列化（网页6的pickle+gzip组合）
        with gzip.open(cache_file, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        # 更新时间戳（网页9的时间记录方法）
        with open(timestamp_file, 'w') as f:
            f.write(datetime.now().isoformat())
        # 7. 触发过期清理（网页10的目录级清理逻辑）
        cleanup_old_cache()
        return result
    return wrapper
####
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
