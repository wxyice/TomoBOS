'''
Data: 2021.11.18
Author: Wxyice
线性方程求解器
关于几种常见的求解器的实现
1. SD
    最速梯度下降法（也就是
2. CG
    共轭梯度下降法
    要求系数矩阵为对称阵
3. Gauss-Seidel
    高斯赛德尔迭代
    要求系数矩阵严格的对角占优
4. ART

'''


from tqdm import tqdm

try:
    # use gpu accelerate
    import cupy as np
    np.random.random((10,10))
    device='gpu'
except:
    # use cpu accelerate
    import numba
    import numpy as np
    device='cpu'


def SD(A,b,epsilon=1e-5,maxiter=1e3):
    '''

    '''
    x=np.zeros(b.shape)

    iter=0
    r=b-np.dot(A,x)
    err=np.sum(r**2)
    while err>epsilon and iter<maxiter:
        iter+=1
        alpha=np.dot(r,r)/np.dot(np.dot(r,A),r)
        x=x+alpha*r
        r=b-np.dot(A,x)
        err=np.sum(r**2)
        print('{0:6d} err={1}'.format(iter,err))
    return x

def Gauss_Seidel(A,b,epsilon=1e-5,maxiter=1e5):
    iter=0
    n=len(b)
    x=np.zeros(b.shape)
    p=x.copy()
    err=10000
    while iter<maxiter and err>epsilon:
        for j in range(n):
            if j==0:
                x[0]=(b[0]-A[0,1:n] @ p[1:n])/A[0,0]
            elif j == n - 1:
                x[n - 1] = (b[n - 1] - (A[n - 1, 0:n - 1] @ x[0:n - 1])) / A[n - 1, n - 1]
            else:
                x[j] = (b[j] - A[j, 0:j] @ x[0:j] - A[j, j + 1:n] @ p[j + 1:n]) / A[j, j]
        err = np.abs(np.sqrt(np.sum((x- p)**2)))
        p=x.copy()
        iter=iter+1
        print('{0:6d} err={1}'.format(iter,err))
    return x


def CG(A, b, epsilon=1e-20, maxiter=1000,device=device):
    '''
        共轭梯度求解
        已经做过gpu计算优化了
    '''
    iter = 0
    if device=='gpu':
        A=np.asarray(A)
        b=np.asarray(b)
    x = np.random.random(b.shape)
    r = b-A @ x
    d = r.copy()
    err = np.sum(r**2)

    # for accelerate the speed
    # the operator @ has the same function as np.dot()
    if device=='cpu':
        @numba.jit(nopython=True, cache=True)
        def loop(A, x, d, r):
            alpha = np.dot(r, r)/(np.dot(np.dot(A, d), d))
            x = x+alpha*d
            r1 = r-alpha*np.dot(A, d)
            beta = np.dot(r1, r1)/np.dot(r, r)
            d = r1+beta*d
            return x, err, d, r1
    else:
        def loop(A, x, d, r):
            alpha = np.dot(r, r)/(np.dot(np.dot(A, d), d))
            x = x+alpha*d
            r1 = r-alpha*np.dot(A, d)
            beta = np.dot(r1, r1)/np.dot(r, r)
            d = r1+beta*d
            return x, err, d, r1

    if device=='cpu':
        bar=tqdm(range(maxiter))
    else:
        bar=range(maxiter)
        
    for iter in bar:
        x, err, d, r = loop(A, x, d, r)
        err = np.sum(r**2)
        if err < epsilon:
            break
    if iter == maxiter-1:
        print('Non convergence')
        #print('{0:<6d} err={1:.8e}'.format(iter,err))
    if device=='gpu':
        x=np.asnumpy(x)
    return x


def CG_cuda(A,b,epsilon=1e-20,maxiter=1000):
    '''
        尝试使用批处理的方式对整个三维矩阵进行计算
        两种方法
        切片模式和直接求解三维的泊松方程。直接求解三维泊松方程指定有点毛病，要改理论
        切片和batch的类似就可以
    '''
    pass

# def CG(A,b,epsilon=1e-20,maxiter=1e3):
#     iter=0
#     x=np.random.random(b.shape)
#     r=b-np.matmul(A,x)
#     d=r.copy()
#     err=np.sum(r**2)

#     while err>epsilon and iter <maxiter:
#         iter+=1
#         alpha=np.dot(r,r)/(np.dot(np.dot(A,d),d))
#         x=x+alpha*d
#         r1=r-alpha*np.dot(A,d)
#         beta=np.dot(r1,r1)/np.dot(r,r)
#         d=r1+beta*d
#         r=r1
#         #err=np.sum(np.sqrt(r**2))
#         err=np.sum(r**2)
#         print('{0:<6d} err={1:.8e}'.format(iter,err))
    
#     return x

def ART(A,b,epsilon,maxiter):
    pass

if __name__ == '__main__':

    A_3=[
        [1.,2,1], 
        [2,5,3], 
        [0,1,9]
    ]

    b_3=[5,2,9]
   

    A_10 = np.array([                                         
    [1.1, 1., 0.,  0.,  0.,  0,  0.,  0.,  0.,  0], 
    [1., 2, 1. , 0.,  0,   0., 0.,  0.,  0.,  0], 
    [ 0., 1., 2, 1., 0.,  0., 0.,  0.,  0.,  0], 
    [ 0.,  0., 1., 2, 1., 0., 0.,  0. , 0.,  0], 
    [ 0.,  0.,  0., 1.,  2, 1.,0.,   0.,  0.,  0], 
    [ 0.,  0.,  0.,  0.,  1., 2,1., 0.,  0.,  0], 
    [ 0.,  0. , 0. , 0. , 0., 1., 2, 1., 0., 0], 
    [ 0.,  0.,  0. , 0. , 0., 0.,  1., 2,  1.,0], 
    [ 0.,  0. , 0.,  0,  0.,  0.,  0.,  1.,2, 1.], 
    [ 0.,  0. , 0. , 0, 0.,   0.,  0.,  0.,  1.,1.1], 
    ])
    b_10 = np.array([2., 3., 3.,3., 3.,3., 3., 3., 3., 2.])       



    A=np.array(A_10)
    b=np.array(b_10)

    #x=Guass_Seidel(A,b)
    #x=SD(A,b)
    import timeit
    start=timeit.default_timer()
    x1=CG(A,b)
    end=timeit.default_timer()
    print(end-start)
    print(x1)

    start=timeit.default_timer()
    x1=CG(A,b)
    end=timeit.default_timer()
    print(end-start)

    #print(x,x1)

    print(x1)



    '''
    n = 6   #其他几个矩阵，测试用
    A = np.array([
    [-3., 1., 0. , 0. ,  0., 0.5 ], 
    [1., -3., 1. , 0. , 0,   0. ], 
    [ 0., 1.,-3., 1., 0. ,   0. ], 
    [ 0.,  0. , 1., -3., 1.,   0.], 
    [ 0.,  0. , 0. , 1.,-3.,  1.], 
    [ 0.5,  0. , 0. , 0. , 1.,  -3.], 
    ]) 
    b = np.array([2.5, 1.5, 1., 1., 1.5, 2.5])

    n = 3
    A = np.array([
    [2.,0.,1.],
    [0.,1.,0.],
    [1.,0.,2.]
    ]) 
    b = np.array([3.,1.,3.])
    '''
