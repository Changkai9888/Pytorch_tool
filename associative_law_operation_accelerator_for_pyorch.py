import torch,time,fc
from torch.cuda.amp import autocast
def get_Normalization(a,dim):
    '''归一化：数据形状要求: batch*行 *列；
    dim=0，直接返回；dim=1，行归一；dim=-1，列归一'''
    if dim!=0:
        if dim==-1:#列归一化
            '''if len(a.shape)==2:#单个矩阵,不是最后的输出，中途的归一化。
                sum_a=torch.sum(a,dim=0)
                a=a/sum_a
            else:#batch序列'''
            sum_a=torch.sum(a,dim=1)
            a=a/(sum_a.unsqueeze(1).expand(a.shape))
        elif dim==1:#行归一化
            '''if len(a.shape)==2:#单个矩阵,不是最后的输出，中途的归一化。
                sum_a=torch.sum(a,dim=1)
                a=a/sum_a.unsqueeze(1)
            else:'''
            sum_a=torch.sum(a,dim=2)
            a=a/sum_a.unsqueeze(2)
        if torch.isnan(sum_a).any() or torch.isinf(sum_a).any():#判断是否有Nan或者inf的坏点
            print(sum_a.dtype)
            print(sum_a)
            fc.save_temp(sum_a,'sum_a')
            if len(a.shape)>2:
                fc.plot(sum_a[:,0])
            stop;
    return a
#f: 1), [n_old]*[n_new]    2), 1_old*[n_new]    3), 1_old*1_new
#例如：选择穿透
def f_gather(old,new):#递归切片
    if old.shape==new.shape and len(old.shape)==2:
        return torch.gather(new, 1, old)
    elif old.shape!=new.shape:
        old=old.repeat(len(new),1)
        return torch.gather(new, 1, old)
    else:
        return torch.gather(new.unsqueeze(0), 1, old.unsqueeze(0)).squeeze(0)

def associa(a,f=1,Normalization=0):
    '''输入序列a=[A1,A2,A3……A(i)]（比如一个矩阵序列），输出为a的累次结合律二元运算序列：
    return=[A1,A1*A2,A1*A2*A3,……,A1*A2*A3*……A(i-1)*A(i)]；
    上式中，*代表符合结合律的双元算符，参数f表示这个算符函数（特别的，在f=1时表示矩阵乘法）；
    Normalization：0，不归一化；-1，列归一化'''
    #print(a)
    if f==1:
        f=lambda old,new: old@new
    elif f==-1:
        f=lambda old,new: new@old
        
    N=len(a);rec=N
    for i in range(int((N/2)**0.5),int(N**0.5+1)):
        _1,_2=int(N/i),i
        _3=N-_1*_2+_1+_2
        if _3<rec:
            num1,num2,rec=_1,_2,_3
    #print(num1,num2,num1*num2)
    sum_a=a[:num1*num2].reshape([num1,num2]+list(a.shape)[1:])
    result=sum_a[:,0].unsqueeze(1)
    for i in range(1,len(sum_a[0])):#多行并行，横向复制
        result=torch.cat((result,f(result[:,i-1],sum_a[:,i]).unsqueeze(1)),dim=1)#见下面，注释1
    result=torch.cat( [result[:,:-1],get_Normalization(result[:,-1],dim=Normalization).unsqueeze(1)] ,dim=1)#过滤一下归一化问题
    result_1=result[0,:].unsqueeze(0)
    for i in range(1,len(sum_a)):#逐行，追尾复制
        result_1=torch.cat((result_1,f(result_1[i-1,-1],result[i,:]).unsqueeze(0)),dim=0)
    sum_a=result_1.reshape([num1*num2]+list(a.shape)[1:])
    for i in range(num1*num2,len(a)):
        sum_a=torch.cat((sum_a,f(sum_a[i-1],a[i]).unsqueeze(0)),dim=0)
    sum_a=get_Normalization(sum_a,dim=Normalization)
    #注释1：这么写梯度可以正常传播，可能是由于torch的某种机制，cat可以传播梯度，而其他某些操作不可以
    return sum_a

#归一性测试
if  __name__=='__main__':
    for i in range(1):
        a=torch.rand(200256,3,3).cuda().to(torch.float32)
        a[:,:,1]=0
        a[:,:,2]=1-a[:,:,0]#行归一
    result=associa(a,Normalization=1)
    print(f'运行结果:  32f，并行 cuda，归一化\n{result[-1]}')#比较运行结果
    print(f'中途结果{int(len(a)*0.5)}运行结果:   16f，并行 cuda，归一化\n{result[int(len(a)*0.5)+13]}')
    a=a.to(torch.float16)
    result=associa(a,Normalization=1)
    print(f'运行结果:  16f，并行 cuda，归一化\n{result[-1]}')#比较运行结果
    print(f'中途结果{int(len(a)*0.5)}运行结果:   32f，并行 cuda，归一化\n{result[int(len(a)*0.5)+13]}')

#速度测试的例子:
"""注意：对于cuda，测试速度前，确保调用运行一遍associa()，
以获得cuda对函数的初始化，这会明显的影响计算速度。"""
if __name__=='__main__':
    k=200253
    compare={};
    _=associa(torch.rand(15,2,2).cuda())#初始化associa
    for i in range(1):#cuda
        a=torch.rand(k,3,3).cuda()#.to(torch.float16)
        a[:,:,1]=0
        a[:,:,2]=1-a[:,:,0]#行归一
        ####
        tim=time.time()
        result=associa(a,Normalization=1)
        compare['并行 cuda ']=time.time()-tim
        re=torch.clone(a)
        re[0]=a[0]
        tim=time.time()
        for i in range(1,len(a)):
            re[i]=re[i-1]@a[i]
        compare['串行 cuda ']=time.time()-tim
    print(f'比较运行结果:  并行 cuda，串行 cuda \n{result[-1]} \n{re[-1]}')#比较运行结果
    print(f'比较中途{int(len(a)*0.5)}运行结果:  并行 cuda，串行 cuda \n{result[int(len(a)*0.5)+13]}\n{re[int(len(a)*0.5)+13]}')#比较中途运行结果
    for i in range(1):#cpu
        a=torch.rand(k,3,3).cpu()#.to(torch.float32)
        a[:,:,1]=0
        a[:,:,2]=1-a[:,:,0]#行归一
        ####
        tim=time.time()
        result=associa(a,Normalization=1)
        compare['并行 cpu ']=time.time()-tim
        re=torch.clone(a)
        re[0]=a[0]
        tim=time.time()
        for i in range(1,len(a)):
            re[i]=re[i-1]@a[i]
        compare['串行 cpu ']=time.time()-tim

    print(f'比较运行结果:  并行 cpu，串行 cpu \n{result[-1]}\n{re[-1]}')#比较运行结果
    print(f'比较中途{int(len(a)*0.5)}运行结果:  并行 cpu，串行 cpu  \n{result[int(len(a)*0.5)+13]}\n{re[int(len(a)*0.5)+13]}')#比较中途运行结果
    print(f'比较时间：\n{compare}')#比较时间
#cpu/cuda, 16/32f, 归一化，串/并行，分割方法 -->时间，结果
