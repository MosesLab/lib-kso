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


ksz = 9


q1 = np.empty([1024, 1024, 1024], dtype=np.float32)
q2 = np.empty([1024, 1024, 1024], dtype=np.float32)
q3 = np.empty([1024, 1024, 1024], dtype=np.float32)

print('Cuda Test')
cuda_start = time.time()
cuda_data = dspk_cuda.denoise_fits_file_quartiles(q1, q2, q3, ksz)
cuda_end = time.time()
cuda_elapsed = cuda_end - cuda_start
print(cuda_elapsed)

q1.resize(cuda_data.shape)
q2.resize(cuda_data.shape)
q3.resize(cuda_data.shape)

q1_flat = q1.flatten()
q2_flat = q2.flatten()
q3_flat = q3.flatten()

q1_flat = q1_flat[q2_flat > 0]
q3_flat = q3_flat[q2_flat > 0]
q2_flat = q2_flat[q2_flat > 0]

# Create linear regression object
regr = linear_model.LinearRegression()

regr.fit(np.expand_dims(q2_flat, -1), np.expand_dims(q3_flat, -1))

x = np.arange(0, 200,1,dtype=np.float32)
y = regr.predict(np.expand_dims(x, -1))
y = np.squeeze(y)

plt.figure()
plt.hist2d(q2_flat, q3_flat, bins=100, norm=colors.SymLogNorm(linthresh=1))
plt.plot(x,y)

plt.figure()
plt.hist2d(q2_flat, q1_flat, bins=100, norm=colors.SymLogNorm(linthresh=1))



fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(ax, q3, 0)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)




plt.show()