import numpy as np
import matplotlib.pyplot as plt
import os

# import dspk_util
#
# from dspk import dspk
# # from dspk_idl import dspk_idl


import kso.tf.img.dspk_tf as dspk_tf
# import kso.cpp.img.dspk.dspk as dspk_cpp
import kso.cuda.img.dspk as dspk_cuda

import kso.py.img.shaping.shaping as dspk_util

from kso.py.tools.scroll_stepper import IndexTracker

import time

print(os.getcwd())

path = ""

pix_dev = 2.0
ksz = 5
Niter = 20

print('Cuda Test')
cuda_start = time.time()
cuda_data = dspk_cuda.denoise_fits_file(path, pix_dev, ksz, Niter)
cuda_end = time.time()
cuda_elapsed = cuda_end - cuda_start
print(cuda_elapsed)


fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(ax, cuda_data, 0, v_min=0, v_max=1000)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)




plt.show()