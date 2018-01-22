import numpy as np
import matplotlib.pyplot as plt

# import dspk_util
#
# from dspk import dspk
# # from dspk_idl import dspk_idl

import kso.cpp.img.dspk.dspk as dspk_cpp

import time

sz_t = 64
sz_y = 64
sz_l = 64

sz = sz_t * sz_y * sz_l


spk_frac = 0.05
# spk_frac = 0.0
n_spk = int(spk_frac * sz)

Niter = 20

noise_mean = 64
spike_mean = 512

pix_dev = 3.0

plt_dev = 2
plt_min = 0
# plt_min = noise_mean - plt_dev * np.sqrt(noise_mean
plt_max = spike_mean + plt_dev * np.sqrt(spike_mean)

# rand = np.random.RandomState(seed=1)
rand = np.random.RandomState(seed=None)


# Initialize background with noise
orig_data = rand.poisson(lam=64, size=[sz_t, sz_y, sz_l])
orig_data = orig_data.astype(np.float32)


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

print(orig_data.strides)

dspk_cpp.remove_noise_3D(orig_data, pix_dev, 5, Niter)


# # Test despiking routine
# tf_start = time.time()
# (dspk_data,good_map,bad_pix_number) = dspk(orig_data, std_dev=pix_dev, Niter=Niter)
# tf_end = time.time()
# tf_elapsed = tf_end - tf_start

# # Compare against IDL despiking routine
# idl_start = time.time()
# idl_data = dspk_idl(orig_data, std_dev=pix_dev, Niter=Niter)
# idl_end = time.time()
# idl_elapsed = idl_end - idl_start
#
# print('tensorflow time =', tf_elapsed)
# print('IDL time =', idl_elapsed)
# print('ratio =', idl_elapsed / tf_elapsed)
#
#
# # Flatten cube so we can view as image
# orig_data_flat = dspk_util.flatten_cube(orig_data[:,:,0:9], sz_t, sz_y, 9)
# good_map_flat = dspk_util.flatten_cube(good_map[:,:,0:9], sz_t, sz_y, 9)
# dspk_data_flat = dspk_util.flatten_cube(dspk_data[:,:,0:9], sz_t, sz_y, 9)
# idl_data_flat = dspk_util.flatten_cube(idl_data[:,:,0:9], sz_t, sz_y, 9)
# # dspk_data_flat = dspk_util.flatten_cube(dspk_data, 9, 9, 9)
# # idl_data_flat = dspk_util.flatten_cube(idl_data, 9, 9, 9)
#
# f1 = plt.figure()
# plt.imshow(orig_data_flat,vmin=plt_min, vmax=plt_max)
#
# # f2 = plt.figure()
# # plt.imshow(good_map_flat)
#
# f3 = plt.figure()
# plt.imshow(dspk_data_flat, vmin=plt_min, vmax=plt_max)
# # plt.imshow(dspk_data_flat)
#
# f4 = plt.figure()
# plt.imshow(idl_data_flat, vmin=plt_min, vmax=plt_max)
# # plt.imshow(idl_data_flat)
#
# diff = idl_data_flat - dspk_data_flat
# f5 = plt.figure()
# plt.imshow(diff)
#
#
#
#
# plt.show()