import torch,time
def f(a,b):
    return a@b
def associa(a,f=f):
    '''输入矩阵序列a=[A1,A2,A3……A(i)]，输出为a的累次结合律二元运算序列result=[A1,A1*A2,A1*A2*A3,……,A1*A2*A3*……A(i-1)*A(i)]
    注意：f为结合律双元算符'''
    n=int(len(a)**0.5)
    i=0
    sum_a=[]
    while n*(i+1)<=len(a):
        sum_a+=[a[n*i:n*(i+1)].unsqueeze(0)]
        i+=1
    size=[n*i]+list(a.shape)[1:]
    sum_a=torch.cat(sum_a,dim=0)
    result=torch.clone(a)
    for i in range(1,len(sum_a[0])):
        sum_a[:,i]=f(sum_a[:,i-1],sum_a[:,i])
    for i in range(1,len(sum_a)):
        sum_a[i,:]=f(sum_a[i-1,-1],sum_a[i,:])
    result[:size[0]]=sum_a.reshape(size)
    for i in range(n**2,len(a)):
        result[i]=f(result[i-1],a[i])
    return result
"""
#速度测试的例子
tim1=[];tim2=[]
for i in range(1):
    a=torch.rand(200000,2,2).cuda()
    tim=time.time()
    result=associa(a,f)
    tim1+=[time.time()-tim]

    tim=time.time()
    re=torch.clone(a)
    re[0]=a[0]
    for i in range(1,len(a)):
        re[i]=re[i-1]@a[i]
    tim2+=[time.time()-tim]
for i in range(1):
    a=torch.rand(200000,2,2)
    tim=time.time()
    result=associa(a,f)
    tim1+=[time.time()-tim]

    tim=time.time()
    re=torch.clone(a)
    re[0]=a[0]
    for i in range(1,len(a)):
        re[i]=re[i-1]@a[i]
    tim2+=[time.time()-tim]

result[-1]==re[-1]
print(result[-1],re[-1])
print(result[int(len(a)**0.5)+13],re[int(len(a)**0.5)+13])
print([tim1,tim2])
"""
