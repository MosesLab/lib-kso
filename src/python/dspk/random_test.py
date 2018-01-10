import numpy as np
import matplotlib.pyplot as plt

import dspk_util

from dspk import dspk

sz_x = 32
sz_y = 32
sz_z = np.square(3)  # must be a square!!

slice_z = 0

n_spk = 512

noise_mean = 64
spike_mean = 512

plt_dev = 2
plt_min = 0
# plt_min = noise_mean - plt_dev * np.sqrt(noise_mean)
plt_max = spike_mean + plt_dev * np.sqrt(spike_mean)

rand = np.random.RandomState(seed=None)

# Initialize background with noise
orig_data = rand.poisson(lam=64, size=[sz_x,sz_y,sz_z])

# Put frames around data for easier viewing
frame = 1
orig_data = dspk_util.add_frame(orig_data, [0, 1], f_sz=frame)

# Add random spikes
spike_mask = np.zeros([sz_x,sz_y,sz_z], dtype=np.int32)
for i in range(n_spk):

    # Loop to make sure we select a new coordinate
    while True:

        # Select random coordinate
        x = rand.randint(frame, sz_x-frame, 1)
        y = rand.randint(frame, sz_y-frame, 1)
        z = rand.randint(frame, sz_z-frame, 1)

        # Check if coordinate has been selected before
        if spike_mask[x,y,z] == 0:
            break



    # Apply spike
    spike_mask[x,y,z] = 1
    orig_data[x,y,z] = rand.poisson(lam=512, size=1)




# Test despiking routine
(dspk_data,good_map,bad_pix_number) = dspk(orig_data, std_dev=2.0)

# Flatten cube so we can view as image
orig_data_flat = dspk_util.flatten_cube(orig_data, sz_x, sz_y, sz_z)
good_map_flat = dspk_util.flatten_cube(good_map, sz_x, sz_y, sz_z)
dspk_data_flat = dspk_util.flatten_cube(dspk_data, sz_x, sz_y, sz_z)

f1 = plt.figure()
plt.imshow(orig_data_flat,vmin=plt_min, vmax=plt_max)

# f2 = plt.figure()
# plt.imshow(good_map_flat)

f3 = plt.figure()
plt.imshow(dspk_data_flat, vmin=plt_min, vmax=plt_max)

# f3 = plt.figure()
# plt.imshow(orig_data[:,:,slice_z], vmin=plt_min, vmax=plt_max)



plt.show()

# try:
#     plt.ion()
#     plt.show()
#     while True:
#
#         pass
#
# # plt.waitforbuttonpress()
# except KeyboardInterrupt:
#     plt.close()