from matplotlib import pyplot as plt
from bos import BOS
import os
import numpy as np


ZD = 0.35
ZA = 0.35
f = 0.10
n0 = 1.00029
rho0=1.293

# initialize the BOS class
Bos_pipeline = BOS(ZD, ZA, f, n0)

n=np.load(r'test\20211129 223633\result\n_3D.npy')
rho=np.load(r'test\20211129 223633\result\rho_3D.npy')

i=10

# Bos_pipeline.show_line_plt(n)
# Bos_pipeline.show_slice_in_3D(n)

#n=n[:-15,:]


fig, ax = plt.subplots(1,1)
#ax.imshow(n, cmap='gray')

n=np.flip(n,axis=0)

X=np.arange(0,n.shape[1])
Y=np.arange(0,n.shape[0])

X,Y=np.meshgrid(X,Y)



ax.contourf(X,Y,n,10)
ax.contour(X,Y,n,10,colors='k')

# h=plt.contourf(n)
# cb=plt.colorbar(h)
# cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。

# # ax[1].xticks(fontsize=16)
# # ax[1].yticks(fontsize=16)
# # ax[1].imshow(rho,cmap='gray')
# # ax[0].set_title('n')
# # ax[1].set_title('rho')

# font = {'family' : 'serif',
#         'color'  : 'darkred',
#         'weight' : 'normal',
#         'size'   : 16,
#         }
# cb.set_label('colorbar',fontdict=font) #设置colorbar的标签字体及其大小
#plt.savefig(os.path.join(path_for_plt, 'n_rho.jpg'))
plt.show()

