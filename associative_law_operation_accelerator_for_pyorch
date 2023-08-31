import torch,time
def get_Normalization(a,dim):
    "归一化：数据形状要求: batch*行 *列"
    if dim==-1:#列归一化
        a=a/(torch.sum(a,dim=1).unsqueeze(1).expand(a.shape))
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
    '''输入矩阵序列a=[A1,A2,A3……A(i)]，输出为a的累次结合律二元运算序列result=[A1,A1*A2,A1*A2*A3,……,A1*A2*A3*……A(i-1)*A(i)]
    注意：f为结合律双元算符
    Normalization：0，不归一化；-1，列归一化'''
    if f==1:
        f=lambda old,new: old@new
    elif f==-1:
        f=lambda old,new: new@old
    n=int(len(a)**0.5)
    i=0
    sum_a=[]
    while n*(i+1)<=len(a):
        sum_a+=[a[n*i:n*(i+1)].unsqueeze(0)]
        i+=1
    size=[n*i]+list(a.shape)[1:]
    sum_a=torch.cat(sum_a,dim=0)
    result=sum_a[:,0].unsqueeze(1)
    for i in range(1,len(sum_a[0])):#多行并行，横向复制
        result=torch.cat((result,f(result[:,i-1],sum_a[:,i]).unsqueeze(1)),dim=1)
    result_1=result[0,:].unsqueeze(0)
    for i in range(1,len(sum_a)):
        result_1=torch.cat((result_1,f(result_1[i-1,-1],result[i,:]).unsqueeze(0)),dim=0)
    sum_a=result_1.reshape(size)
    for i in range(size[0],len(a)):
        sum_a=torch.cat((sum_a,f(sum_a[i-1],a[i]).unsqueeze(0)),dim=0)
    sum_a=get_Normalization(sum_a,dim=Normalization)
    return sum_a

#速度测试的例子
#torch.set_default_dtype(torch.float16)
'''tim1=[];tim2=[]
for i in range(1):
    a=torch.rand(400,2,2).cuda()
    tim=time.time()
    result=associa(a)
    tim1+=[time.time()-tim]
    tim=time.time()
    re=torch.clone(a)
    re[0]=a[0]
    for i in range(1,len(a)):
        re[i]=re[i-1]@a[i]
    tim2+=[time.time()-tim]
for i in range(1):
    a=torch.rand(400,2,2)
    tim=time.time()
    result=associa(a)
    tim1+=[time.time()-tim]
    tim=time.time()
    re=torch.clone(a)
    re[0]=a[0]
    for i in range(1,len(a)):
        re[i]=re[i-1]@a[i]
    tim2+=[time.time()-tim]

result[-1]==re[-1]
print(result[-1],re[-1])#比较运行结果
print(result[int(len(a)**0.5)+13],re[int(len(a)**0.5)+13])
print([tim1,tim2])#比较时间'''


