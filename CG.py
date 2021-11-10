import numpy as np

def y(x):
    return x**2-3*x+1

def SD(A,b,epsilon=1e-5):
    x=np.zeros(b.shape)

    iter=0
    r=b-np.matmul(A,x)
    rr=np.dot(r,r)
    while np.sqrt(rr)>epsilon:
        iter=iter+1
        Ar=np.matmul(A,r)
        alpha=rr/np.dot(r,Ar)
        x=x+alpha*r
        r=r-alpha*Ar
        rr=np.dot(r,r)
        print('{0:6d} x={1}'.format(iter,x))
    return x

def CG(A,b,epsilon=1e-5):
    iter=0
    x=np.random.random(b.shape)
    err=b-np.matmul(A,x)
    rr_



A=[
    [1,2,5], 
    [2,5,3], 
    [0,1,9]
]

b=[5,2,9]

A=np.array(A)
b=np.array(b)

x=SD(A,b)

print(x)