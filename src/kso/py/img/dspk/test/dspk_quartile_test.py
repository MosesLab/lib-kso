import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

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


ksz = 25


q2_3 = np.empty([1024, 1024, 1024], dtype=np.float32)

hsx = 1024
hsy = 4096
ndim = 3
hist_3 = np.empty([ndim, hsy, hsx], dtype=np.float32)
cumsum_3 = np.empty([ndim, hsy, hsx], dtype=np.float32)
t0_3 = np.empty([ndim, hsx], dtype=np.float32)
t1_3 = np.empty([ndim, hsx], dtype=np.float32)


print('Cuda Test')
cuda_start = time.time()
dt = dspk_cuda.denoise_fits_file_quartiles(q2_3,  hist_3, cumsum_3, t0_3, t1_3, hsx, hsy, ksz)
cuda_end = time.time()
cuda_elapsed = cuda_end - cuda_start
print(cuda_elapsed)

dt_flat = dt.flatten()
q2_3.resize((ndim,) + dt.shape)
print(q2_3.shape)

for ax in range(0,ndim):


    q2 = q2_3[ax,:, :, :]
    hist = hist_3[ax, :, :]
    cumsum = cumsum_3[ax, :, :]
    t0 = t0_3[ax, :]
    t1 = t1_3[ax, :]

    q2_flat = q2.flatten()



    # plt.figure()
    # plt.hist2d(q2_flat, dt_flat, bins=[500,4096], norm=colors.SymLogNorm(linthresh=1))
    # plt.colorbar()

    plt.figure()
    plt.imshow(hist, norm=colors.SymLogNorm(linthresh=1), origin='lower')
    plt.plot(t0)
    plt.plot(t1)
    #
    # plt.figure()
    # plt.imshow(q2_3[ax, 53,])
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(cumsum, norm=colors.SymLogNorm(linthresh=1))
    # plt.colorbar()


fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(ax, dt, 0)
# tracker = IndexTracker(ax, q2_3[0,], 0)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)




plt.show()