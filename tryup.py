import numba
from numba import jit
from numba.np.ufunc import parallel


import numpy as np
import timeit

@jit(nopython=True)
def main(a):
    s=0
    for i in range(1000000):
        s=s+a
    return s

@numba.jit(nopython=True,cache=True)
def mat(A,x):
    return np.dot(A,x)

def mat0(A,x):
    return np.dot(A,x)
    
A=np.ones((10,10))
x=np.ones((1,10))


s=timeit.default_timer()
a=mat0(A,x.T)
e=timeit.default_timer()

print(e-s)

s=timeit.default_timer()
a=mat(A,x.T)
e=timeit.default_timer()

print(e-s)

s=timeit.default_timer()
a=mat0(A,x.T)
e=timeit.default_timer()

print(e-s)

s=timeit.default_timer()
a=mat(A,x.T)
e=timeit.default_timer()

print(e-s)

