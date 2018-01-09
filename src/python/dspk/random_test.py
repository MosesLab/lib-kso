import numpy as np
import matplotlib.pyplot as plt

import time

import random

from dspk import dspk

sz_x = 16
sz_y = 16
sz_z = 3

slice_z = 1

n_spk = 5

#spiky data
data = np.ones([sz_x,sz_y,sz_z], dtype=np.float32)
data = np.random.poisson(data)

for i in range(n_spk):

    x = random.randrange(sz_x)
    y = random.randrange(sz_y)

    data[x,y,slice_z] = 32.0


f1 = plt.figure()
plt.imshow(data[:,:,slice_z])

(data,good_map,bad_pix_number) = dspk(data)

print(data.shape)
print(good_map.shape)

print(np.where(good_map==0))
print(data[np.where(good_map==0)])



f2 = plt.figure()
plt.imshow(good_map[:,:,slice_z])

f3 = plt.figure()
plt.imshow(data[:,:,slice_z])

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