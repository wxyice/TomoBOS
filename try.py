import numpy as np
from fealpy.mesh import MeshFactory
import matplotlib.pyplot as plt
# %matplotlib inline

# 加载pde模型
pde = possion_solution()

# 加载网格
mf = MeshFactory()
box = pde.domain
mesh = mf.boxmesh2d(box, nx = 10, ny = 10, meshtype = 'quad')

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