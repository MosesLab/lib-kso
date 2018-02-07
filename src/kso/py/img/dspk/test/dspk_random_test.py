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

sz_t = 256
sz_y = 256
sz_l = 256

sz = sz_t * sz_y * sz_l


spk_frac = 0.05
# spk_frac = 0.0
n_spk = int(spk_frac * sz)

Niter = 5

noise_mean = 64
spike_mean = 512

pix_dev = 3.0

plt_dev = 2
plt_min = 0
# plt_min = noise_mean - plt_dev * np.sqrt(noise_mean
plt_max = spike_mean + plt_dev * np.sqrt(spike_mean)

rand = np.random.RandomState(seed=1)
# rand = np.random.RandomState(seed=None)


# Initialize background with noise
orig_data = rand.poisson(lam=64, size=[sz_t, sz_y, sz_l])
orig_data = orig_data.astype(np.float32)
# orig_data = np.empty([sz_t, sz_y, sz_l], dtype=np.float32)

# Add signal
x = np.arange(0, sz_l, 1, np.float32)
y = np.arange(0, sz_y,1, np.float32)
x = np.expand_dims(x, 0)
x = np.expand_dims(x,0)
y = np.expand_dims(y,0)
y = np.expand_dims(y,-1)
x0 = sz_l // 2
y0 = sz_y // 2
fwhm = sz_l / 5
signal = 512 * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)
print(signal)
orig_data = orig_data + signal

# Add random spikes
coords = np.arange(sz)
rand.shuffle(coords)
coords = coords[:n_spk]
orig_data = orig_data.flatten()
orig_data[coords] = rand.poisson(lam=512, size=n_spk)
orig_data = orig_data.reshape([sz_t, sz_y, sz_l])

# Put frames around data for easier viewing
# frame = 0
# orig_data = dspk_util.add_frame(orig_data, [0, 1], f_sz=frame)





# print('C++ Test')
#
# cpp_start = time.time()
# gm_cpp = dspk_cpp.locate_noise_3D(orig_data, pix_dev, 5, Niter)
# cpp_end = time.time()
# cpp_elapsed = cpp_end - cpp_start
# print(cpp_elapsed)
# gm_cpp_flat = gm_cpp[T,:,:]
# f2 = plt.figure()
# plt.imshow(gm_cpp_flat)


print('Cuda Test')
gm_cuda = np.empty([sz_t, sz_y, sz_l], dtype=np.float32)
cuda_start = time.time()
dspk_cuda.denoise_ndarr(orig_data, gm_cuda, pix_dev, 5, Niter)
cuda_end = time.time()
cuda_elapsed = cuda_end - cuda_start
print(cuda_elapsed)
#
# T = 23
# orig_data_flat = orig_data[T,:,:]
# f1 = plt.figure()
# plt.imshow(orig_data_flat,vmin=plt_min, vmax=plt_max)

fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(ax, orig_data, 0, v_min=plt_min, v_max=plt_max)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)

# gm_cuda_flat = gm_cuda[T,:,:]
# f3 = plt.figure()
# plt.imshow(gm_cuda_flat)


#
# print('Tensorflow Test')
# tf_start = time.time()
# (data_tf,gm_tf,bad_pix_number) = dspk_tf.dspk(orig_data, std_dev=pix_dev, Niter=Niter)
# tf_end = time.time()
# tf_elapsed = tf_end - tf_start
# print(tf_elapsed)
# gm_tf_flat = gm_tf[T,:,:]
# f4 = plt.figure()
# plt.imshow(gm_tf_flat)




# Flatten cube so we can view as image
# orig_data_flat = dspk_util.flatten_cube(orig_data[:,:,0:9], sz_t, sz_y, 9)
# gm_cpp_flat = dspk_util.flatten_cube(gm_cpp[:,:,0:9], sz_t, sz_y, 9)
# gm_cuda_flat = dspk_util.flatten_cube(gm_cuda[:,:,0:9], sz_t, sz_y, 9)
# gm_tf_flat = dspk_util.flatten_cube(gm_tf[:,:,0:9], sz_t, sz_y, 9)

#
# diff = np.sum(gm_cuda - gm_tf, axis=(1,2))
#
# plt.figure()
# plt.plot(diff)
# #
#
# plt.figure()
# plt.imshow(np.sum(gm_cuda - gm_tf, axis=2))


plt.show()