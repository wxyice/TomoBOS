import numpy as np
import os


def grad(x,y,Dx,Dy):
    """
    x,y 是各个控制点的对应真实位置，分别用xy坐标表示，应该来源于标定函数
    u,v 是各个控制点的位移量
    这个函数是实现求导功能
    """    
    W,H=Dx.shape

    G=np.zeros((W,H))
    # 循环求导
    for i in range(1,W-1):
        for j in range(1,H-1):
            Gx=(Dx[i][j+1]-Dx[i][j-1])/(2*(x[i][j+1]-x[i][j]))
            Gy=(Dy[i+1][j]-Dy[i-1][j])/(2*(y[i+1][j]-y[i][j]))
            G[i][j]=Gx+Gy
    # G.shape=W-2,H-2
    return G

def cut_domain(domain):
    return domain[1:-1,1:-1].copy()


def makeA(x,y,Dx,Dy):

    G=grad(x,y,Dx,Dy)
    W,H=Dx.shape
    num=W*H
    A=np.zeros((num,num))
    B=np.zeros((num,1))
    ay=1/4
    ax=1/4
    for i in range(1,W):
        for j in range(1,H):
            line=i*W+j
            B[line]=G[i][j]
            A[line][i*W+j]=1
            A[line][i*W+(j-1)],A[line][i*W+(j+1)]=ay,ay
            A[line][(i+1)*W+j],A[line][(i-1)*W+j]=ax,ax
    
    return A,B

def boundary(A,B,W,H):
    for j in range(H):
        pass
    return A,B


def CG(A,B):
    
    




if __name__ == '__main__':
    """
    注意：x是竖直方向，y是水平方向
    """

    x=[
        [0,1,2,3,4,5,6],
        [0,1,2,3,4,5,6],
        [0,1,2,3,4,5,6],
        [0,1,2,3,4,5,6],
        [0,1,2,3,4,5,6]
    ]
    y=[
        [0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1], 
        [2,2,2,2,2,2,2],
        [3,3,3,3,3,3,3],
        [4,4,4,4,4,4,4]
    ]
    test_x=[
        [1,2,3,4,5,6,7], 
        [3,4,5,7,8,2,5], 
        [4,5,8,1,5,9,2], 
        [2,4,8,1,8,0,2], 
        [1,4,5,7,8,6,1]
    ]
    test_y=[
        [7,2,3,4,0,6,5], 
        [3,3,4,7,8,2,5], 
        [4,5,8,6,2,9,2], 
        [2,4,8,6,7,8,2], 
        [3,9,5,6,8,1,1]
    ]
    x=np.array(x)
    y=np.array(y)
    test_x=np.array(test_x)
    test_y=np.array(test_y)

    print(test_x.shape)
    G=grad(x,y,test_x,test_y)
    print(G)
    B=G.reshape(-1)
    print(B.shape)

    
