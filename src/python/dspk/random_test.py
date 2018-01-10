import numpy as np
import matplotlib.pyplot as plt

from dspk_util import flatten_cube

from dspk import dspk

sz_x = 32
sz_y = 32
sz_z = np.square(2)  # must be a square!!

slice_z = 0

n_spk = 512

noise_mean = 64
spike_mean = 512

plt_dev = 2
plt_min = noise_mean - plt_dev * np.sqrt(noise_mean)
plt_max = spike_mean + plt_dev * np.sqrt(spike_mean)

rand = np.random.RandomState(seed=None)

# Initialize background with noise
orig_data = rand.poisson(lam=64, size=[sz_x,sz_y,sz_z])

# Put frames around data for easier viewing
# orig_data[0,:,:] = 0
# orig_data[sz_x -1,:,:] = 0
# orig_data[:,0,:] = 0
# orig_data[:,sz_y - 1,:] = 0

# Add random spikes
spike_mask = np.zeros([sz_x,sz_y,sz_z], dtype=np.int32)
for i in range(n_spk):

    # Loop to make sure we select a new coordinate
    while True:

        # Select random coordinate
        x = rand.randint(0, sz_x, 1)
        y = rand.randint(0, sz_y, 1)
        z = rand.randint(0, sz_z, 1)

        # Check if coordinate has been selected before
        if spike_mask[x,y,z] == 0:
            break



    # Apply spike
    spike_mask[x,y,z] = 1
    orig_data[x,y,z] = rand.poisson(lam=512, size=1)




# Test despiking routine
(dspk_data,good_map,bad_pix_number) = dspk(orig_data, std_dev=2.0)

# Flatten cube so we can view as image
orig_data_flat = flatten_cube(orig_data, sz_x, sz_y, sz_z)
good_map_flat = flatten_cube(good_map, sz_x, sz_y, sz_z)
dspk_data_flat = flatten_cube(dspk_data, sz_x, sz_y, sz_z)

f1 = plt.figure()
plt.imshow(orig_data_flat,vmin=plt_min, vmax=plt_max)

f2 = plt.figure()
plt.imshow(good_map_flat)

f3 = plt.figure()
plt.imshow(dspk_data_flat, vmin=plt_min, vmax=plt_max)



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