import numpy as np
import matplotlib.pyplot as plt

import dspk_util

from dspk import dspk
from dspk_idl import dspk_idl

sz_x = 32
sz_y = 32
sz_z = np.square(3)  # must be a square!!

sz = sz_x * sz_y * sz_z

slice_z = 0

spk_frac = 0.05
# spk_frac = 0.0
n_spk = int(spk_frac * sz)

Niter = 1

noise_mean = 64
spike_mean = 512

pix_dev = 4.0

plt_dev = 2
plt_min = 0
# plt_min = noise_mean - plt_dev * np.sqrt(noise_mean
plt_max = spike_mean + plt_dev * np.sqrt(spike_mean)

rand = np.random.RandomState(seed=1)
# rand = np.random.RandomState(seed=None)


# Initialize background with noise
orig_data = rand.poisson(lam=64, size=[sz_x,sz_y,sz_z])
# orig_data = np.ones([sz_x,sz_y,sz_z], dtype=np.float32)



# Put frames around data for easier viewing
frame = 0
orig_data = dspk_util.add_frame(orig_data, [0, 1], f_sz=frame)

# Add random spikes
spike_mask = np.zeros([sz_x,sz_y,sz_z], dtype=np.int32)
for i in range(n_spk):

    # Loop to make sure we select a new coordinate
    while True:

        # Select random coordinate
        x = rand.randint(frame, sz_x-frame, 1)
        y = rand.randint(frame, sz_y-frame, 1)
        z = rand.randint(0, sz_z, 1)

        # Check if coordinate has been selected before
        if spike_mask[x,y,z] == 0:
            break



    # Apply spike
    spike_mask[x,y,z] = 1
    orig_data[x,y,z] = rand.poisson(lam=512, size=1)



# Test despiking routine
(dspk_data,good_map,bad_pix_number) = dspk(orig_data, std_dev=pix_dev, Niter=Niter)

# Compare against IDL despiking routine
idl_data = dspk_idl(orig_data, std_dev=pix_dev, Niter=Niter)


# Flatten cube so we can view as image
orig_data_flat = dspk_util.flatten_cube(orig_data, sz_x, sz_y, sz_z)
good_map_flat = dspk_util.flatten_cube(good_map, sz_x, sz_y, sz_z)
dspk_data_flat = dspk_util.flatten_cube(dspk_data, sz_x, sz_y, sz_z)
idl_data_flat = dspk_util.flatten_cube(idl_data, sz_x, sz_y, sz_z)
# dspk_data_flat = dspk_util.flatten_cube(dspk_data, 9, 9, 9)
# idl_data_flat = dspk_util.flatten_cube(idl_data, 9, 9, 9)

f1 = plt.figure()
plt.imshow(orig_data_flat,vmin=plt_min, vmax=plt_max)

# f2 = plt.figure()
# plt.imshow(good_map_flat)

f3 = plt.figure()
plt.imshow(dspk_data_flat, vmin=plt_min, vmax=plt_max)
# plt.imshow(dspk_data_flat)

f4 = plt.figure()
plt.imshow(idl_data_flat, vmin=plt_min, vmax=plt_max)
# plt.imshow(idl_data_flat)

diff = idl_data_flat - dspk_data_flat
f5 = plt.figure()
plt.imshow(diff)




plt.show()
