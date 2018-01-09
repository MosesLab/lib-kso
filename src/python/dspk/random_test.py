import numpy as np
import matplotlib.pyplot as plt

import time

import random

from dspk import dspk

sz_x = 32
sz_y = 32
sz_z = 8

slice_z = 1

n_spk = 512

plt_min = 128
plt_max = 512

#spiky data
# data = np.ones([sz_x,sz_y,sz_z], dtype=np.float32)
data = np.random.poisson(lam=256, size=[sz_x,sz_y,sz_z])

for i in range(n_spk):

    x = random.randrange(sz_x)
    y = random.randrange(sz_y)
    z = random.randrange(sz_z)

    data[x,y,z] = np.random.poisson(lam=512, size=1)


f1 = plt.figure()
plt.imshow(data[:,:,slice_z],vmin=plt_min, vmax=plt_max)

(data,good_map,bad_pix_number) = dspk(data, std_dev=3.0)

f2 = plt.figure()
plt.imshow(good_map[:,:,slice_z])

f3 = plt.figure()
plt.imshow(data[:,:,slice_z],vmin=plt_min, vmax=plt_max)

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