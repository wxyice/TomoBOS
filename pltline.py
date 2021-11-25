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

n=np.load(r'test\20211125 230807\result\n.npy')
rho=np.load(r'test\20211125 230807\result\rho.npy')

i=10

Bos_pipeline.show_line_plt(n)
Bos_pipeline.show_slice_in_3D(n)


fig, ax = plt.subplots(1, 2)
ax[0].imshow(n, cmap='gray')
ax[1].imshow(rho,cmap='gray')
ax[0].set_title('n')
ax[1].set_title('rho')
#plt.savefig(os.path.join(path_for_plt, 'n_rho.jpg'))
plt.show()

