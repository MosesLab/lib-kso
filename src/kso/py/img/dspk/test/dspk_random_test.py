import numpy as np
import matplotlib.pyplot as plt

# import dspk_util
#
# from dspk import dspk
# # from dspk_idl import dspk_idl

import kso.tf.img.dspk_tf as dspk_tf
import kso.cpp.img.dspk.dspk as dspk_cpp
import kso.cuda.img.dspk.dspk_cuda as dspk_cuda

import kso.py.img.shaping.shaping as dspk_util

import time

sz_t = 128
sz_y = 128
sz_l = 128

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

# Put frames around data for easier viewing
# frame = 0
# orig_data = dspk_util.add_frame(orig_data, [0, 1], f_sz=frame)

# Add random spikes
coords = np.arange(sz)
np.random.shuffle(coords)
coords = coords[:n_spk]
orig_data = orig_data.flatten()
orig_data[coords] = rand.poisson(lam=512, size=n_spk)
orig_data = orig_data.reshape([sz_t, sz_y, sz_l])


# cpp_start = time.time()
# gm_cpp = dspk_cpp.locate_noise_3D(orig_data, pix_dev, 5, Niter)
# cpp_end = time.time()
# cpp_elapsed = cpp_end - cpp_start
# print(cpp_elapsed)

cuda_start = time.time()
gm_cuda = dspk_cuda.locate_noise_3D(orig_data, pix_dev, 5, Niter)
cuda_end = time.time()
cuda_elapsed = cuda_end - cuda_start
print(cuda_elapsed)


# Test despiking routine
tf_start = time.time()
(data_tf,gm_tf,bad_pix_number) = dspk_tf.dspk(orig_data, std_dev=pix_dev, Niter=Niter)
tf_end = time.time()
tf_elapsed = tf_end - tf_start
print(tf_elapsed)

# Flatten cube so we can view as image
orig_data_flat = dspk_util.flatten_cube(orig_data[:,:,0:9], sz_t, sz_y, 9)
# gm_cpp_flat = dspk_util.flatten_cube(gm_cpp[:,:,0:9], sz_t, sz_y, 9)
gm_cuda_flat = dspk_util.flatten_cube(gm_cuda[:,:,0:9], sz_t, sz_y, 9)
gm_tf_flat = dspk_util.flatten_cube(gm_tf[:,:,0:9], sz_t, sz_y, 9)


f1 = plt.figure()
plt.imshow(orig_data_flat,vmin=plt_min, vmax=plt_max)

# f2 = plt.figure()
# plt.imshow(gm_cpp_flat)

f3 = plt.figure()
plt.imshow(gm_cuda_flat)

f4 = plt.figure()
plt.imshow(gm_tf_flat)

plt.show()