#https://blog.csdn.net/cium123/article/details/109004730
import sys

import matplotlib.pyplot as plt
import numpy as np
from fealpy.boundarycondition import DirichletBC, NeumannBC
from fealpy.decorator import barycentric, cartesian
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.mesh import MeshFactory
from fealpy.tools.show import show_error_table, showmultirate
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import spsolve


class possion_solution:
    
    def __init__(self):
        pass
    
    # 定义网格区域大小
    @property
    def domain(self):   
        return np.array([0, 0.1, 0, 0.1])
        
    # 原项 f(x,y)，泊松方程的右边
    @cartesian
    def source(self, p):  
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 2*pi*pi*np.cos(pi*x)*np.cos(pi*y)
        return val
    
    # 精确解u(x,y)，泊松方程的左边
    @cartesian
    def exact_solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
        return val
        
    # 真解的梯度 
    @cartesian
    def gradient(self, p):  
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype = np.float64)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val
        
    # 梯度的负方向
    @cartesian 
    def flux(self, p):
        return -self.gradient(p)
        
    #定义边界条件
    @cartesian
    def is_dirichlet_boundary(self, p):
        y = p[..., 1]
        return (y == 0.0) 
    
    @cartesian
    def dirichlet(self, p):
        return self.exact_solution(p)
    
    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return (x == 0.0)|(x == 0.1)|(y == 0.1)
        
    @cartesian
    def neumann(self, p, n):
        grad = self.gradient(p)
        val = np.sum(grad * n, axis = -1)
        return val




def possion_solution_solver(pde, n, refine, order): 
    """
    Input:
        @pde: 定义偏微分方程
        @n: 初始网格剖分段数
        @refine: 网格加密的最大次数（迭代求解次数）
        @order: 有限元多项式次数
    Output: None
    """
    #mf = MeshFactory()
    mesh = MeshFactory.boxmesh2d(pde.domain, nx = n, ny = n, meshtype = 'tri')
    
    number_of_dofs = np.zeros(refine, dtype = mesh.itype)
    # 建立空数组，目的把每组的自由度个数存下来
    error_matrix = np.zeros((2, refine), dtype = mesh.ftype)
    error_type = ['$||u - u^{h}||_{0}$', '$||\\nabla u - \\nabla u^{h}||_{0}$']
    
    for i in range(refine):
        femspace = LagrangeFiniteElementSpace(mesh, p = order)
        number_of_dofs[i] = femspace.number_of_global_dofs()
        uh = femspace.function() 
        # 返回一个有限元函数，初始自由度值全为 0
        
        # A·u = b + b_n
        
        A = femspace.stiff_matrix()
        F = femspace.source_vector(pde.source)
        # 先计算纽曼
        bc = NeumannBC(femspace, pde.neumann, threshold = pde.is_neumann_boundary)
        F = bc.apply(F)
        
        #最后计算Dirichlet
        bc = DirichletBC(femspace, pde.dirichlet, threshold = pde.is_dirichlet_boundary)
        A, F = bc.apply(A, F, uh)
        
        uh[:] = spsolve(A, F)
        
        #计算误差
        error_matrix[0, i] = femspace.integralalg.L2_error(pde.exact_solution, uh.value)
        error_matrix[1, i] = femspace.integralalg.L2_error(pde.gradient, uh.grad_value)
        
        print('插值点: ', femspace.interpolation_points().shape)
        print('自由度数（NDof）: ', number_of_dofs[i])
        nodes = mesh.entity('node')
        print('节点数: ', nodes.shape)
        if i < refine - 1:
            mesh.uniform_refine()
    
    fig = plt.figure()
    axes = fig.gca(projection = '3d')
    uh.add_plot(axes, cmap = 'rainbow')
    showmultirate(plt, 0, number_of_dofs, error_matrix, error_type, propsize = 20)
    show_error_table(number_of_dofs, error_type, error_matrix, f='e', pre=4, sep=' & ', out=sys.stdout, end='\n')
    plt.show()


# %matplotlib inline
if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # 加载pde模型
    pde = possion_solution()

    # 加载网格
    #mf = fealpy.mesh.MeshFactory()
    box = pde.domain
    mesh = MeshFactory.boxmesh2d(box, nx = 100, ny = 100, meshtype = 'quad')

    # 画图
    figure = plt.figure()
    axes = figure.gca()
    mesh.add_plot(axes)
    #mesh.find_node(axes, showindex = True)
    #mesh.find_edge(axes, showindex = True)
    mesh.find_cell(axes, showindex = True)

    #获取单元，网格，节点信息
    nodes = mesh.entity('node')
    cells = mesh.entity('cell')
    edges = mesh.entity('edge')
    possion_solution_solver(possion_solution(), 10, 5, 1)
    possion_solution_solver(possion_solution(), 10, 5, 2)

    # def is_dirichlet_boundary(self, p):
    #         x = p[..., 0]
    #         y = p[..., 1]
    #         return (y == 0.0) & ( x <b) & (x > a) 
        
    #     @cartesian
    #     def dirichlet(self, p):
    #         val = vall * np.ones(len(p))
    #         return val
