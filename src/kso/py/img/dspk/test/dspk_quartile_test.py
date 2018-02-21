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


ksz = 5


q2 = np.empty([1024, 1024, 1024], dtype=np.float32)

hsx = 1024
hsy = 4096
hist = np.empty([hsy, hsx], dtype=np.float32)
cumsum = np.empty([hsy, hsx], dtype=np.float32)
t0 = np.empty([hsx], dtype=np.float32)
t1 = np.empty([hsx], dtype=np.float32)


print('Cuda Test')
cuda_start = time.time()
dt = dspk_cuda.denoise_fits_file_quartiles(q2,  hist, cumsum, t0, t1, hsx, hsy, ksz)
cuda_end = time.time()
cuda_elapsed = cuda_end - cuda_start
print(cuda_elapsed)

# q1.resize(dt.shape)
q2.resize(dt.shape)
# q3.resize(dt.shape)t
dt_flat = dt.flatten()
# q1_flat = q1.flatten()
q2_flat = q2.flatten()
# q3_flat = q3.flatten()

# q1_flat = q1_flat[q2_flat > 0]
# q3_flat = q3_flat[q2_flat > 0]
# q2_flat = q2_flat[q2_flat > 0]

# Create linear regression object
# regr = linear_model.LinearRegression()
# regr.fit(np.expand_dims(q2_flat, -1), np.expand_dims(q3_flat, -1))
# print(regr.coef_)
# print(regr.intercept_)
# x = np.arange(0, 200,1,dtype=np.float32)
# y = regr.predict(np.expand_dims(x, -1))
# y = np.squeeze(y)

# plt.figure()
# plt.hist2d(q2_flat, q3_flat - q2_flat, bins=100, norm=colors.SymLogNorm(linthresh=1))
#
#
# plt.figure()
# plt.hist2d(q2_flat, q1_flat - q2_flat, bins=100, norm=colors.SymLogNorm(linthresh=1))

plt.figure()
plt.hist2d(q2_flat, dt_flat, bins=[500,4096], norm=colors.SymLogNorm(linthresh=1))
plt.colorbar()

plt.figure()
plt.imshow(hist, norm=colors.SymLogNorm(linthresh=1), origin='lower')
plt.plot(t0)
plt.plot(t1)
plt.colorbar()

plt.figure()
plt.imshow(cumsum, norm=colors.SymLogNorm(linthresh=1))
plt.colorbar()


fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(ax, dt, 0)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)




plt.show()