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
from subprocess import call

print(os.getcwd())


ksz = 25


q2_3 = np.empty([1024, 1024, 1024], dtype=np.float32)

hsx = 1024
hsy = 1024
ndim = 3
nmet = ndim + 1
hist_3 = np.empty([nmet, hsy, hsx], dtype=np.float32)
cumsum_3 = np.empty([nmet, hsy, hsx], dtype=np.float32)
t0_3 = np.empty([nmet, hsx], dtype=np.float32)
t1_3 = np.empty([nmet, hsx], dtype=np.float32)

# for histogram axes
dt_max = 16384
dt_min = -200
Dt = dt_max - dt_min
bwx = Dt / (hsx - 1)
bwy = Dt / (hsy - 1)

# get hist X
def ghx(X):
    return X * bwx + dt_min

# get hist X
def ghy(Y):
    return Y * bwy + dt_min


print('Cuda Test')
cuda_start = time.time()
dt = dspk_cuda.denoise_fits_file_quartiles(q2_3,  hist_3, cumsum_3, t0_3, t1_3, hsx, hsy, ksz)
cuda_end = time.time()
cuda_elapsed = cuda_end - cuda_start
print(cuda_elapsed)

odt = dspk_cuda.read_fits_file()

kdt = np.fromfile('/home/byrdie/obs.dat', dtype=np.float32, count=-1)
print(kdt.shape)
kdt = kdt.reshape(dt.shape)
print(kdt.shape)

dt_flat = dt.flatten()
q2_3.resize((nmet,) + dt.shape)
print(q2_3.shape)

for ax in range(0,nmet):


    q2 = q2_3[ax,:, :, :]
    hist = hist_3[ax, :, :]
    cumsum = cumsum_3[ax, :, :]
    t0 = t0_3[ax, :]
    t1 = t1_3[ax, :]

    q2_flat = q2.flatten()





    x0 = ghx(np.arange(0, hsx))
    y0 = ghy(np.arange(0, hsy))

    plt.figure(figsize=(4.0, 4.0))
    im = plt.imshow(hist, norm=colors.SymLogNorm(1e-4), origin='lower', extent=(-200, 16384, -200, 16384))
    plt.plot(x0, ghx(t0), 'w')
    plt.plot(y0, ghy(t1), 'w')
    plt.xlim(0,768)
    plt.ylim(-50,768)
    plt.xlabel('local median intensity (DN)')
    plt.ylabel('intensity (DN)')
    plt.savefig('hist_' + str(ax) + '.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
    # if ax == 2: plt.colorbar(fraction=0.0497, pad=0.04)

    if ax == 0:
        plt.figure(figsize=(1,8))
        a = np.array([[0, 1]])
        img = plt.imshow(a)
        plt.gca().set_visible(False)
        plt.colorbar(im)
        # plt.savefig('cbar.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)




fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(ax, dt, 0, v_min=-5, v_max=256, cmap='gray')
# tracker = IndexTracker(ax, q2_3[0,], 0)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)

def region_1(dt, path):
    plt.figure(figsize=(4.0,4.0))
    cmap = plt.cm.get_cmap('gray')
    cmap.set_bad(color='red')
    # plt.imshow(dt[54,35:135,295:395]+5, vmin=0, vmax=1024, cmap=cmap, norm=colors.PowerNorm(0.5))
    plt.imshow(dt[50, 35:155, 285:405] + 5, vmin=0, vmax=1024, cmap=cmap, norm=colors.PowerNorm(0.5))
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=300)

def region_2(dt, path):
    plt.figure(figsize=(4.0,4.0))
    cmap = plt.cm.get_cmap('gray')
    cmap.set_bad(color='red')
    # plt.imshow(dt[23,230:330,295:395]+5, vmin=0, vmax=1024, cmap=cmap, norm=colors.PowerNorm(0.5))
    plt.imshow(dt[24, 230:350, 285:405] + 5, vmin=0, vmax=1024, cmap=cmap, norm=colors.PowerNorm(0.5))
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    current_cmap = plt.cm.get_cmap()
    current_cmap.set_bad(color='red')
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=300)


region_1(dt, 'dspk_1.pdf')
region_1(odt, 'orig_1.pdf')
region_1(kdt, 'despike_1.pdf')

region_2(dt, 'dspk_2.pdf')
region_2(odt, 'orig_2.pdf')
region_2(kdt, 'despike_2.pdf')

plt.show()